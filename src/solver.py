from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch

import wandb
from pyscipopt import quicksum
from src.net import OptSatGNN
from src.problem import Instance, ModelWithPrimalDualIntegral


@dataclass
class Result:
    infeasible: bool
    runtime: float
    objective: int
    gap: float
    primal_curve: list

class ONTSSolver():
    def __init__(self, timeout=60) -> None:
        self.timeout = timeout

    @abstractmethod
    def solve(self, instance: Instance) -> Result:
        pass

class SCIPSolver(ONTSSolver):
    def solve(self, instance: Instance, hide_output=True) -> Result:
        model = self.load_model(instance, hide_output=hide_output)

        model.optimize()

        result = self.get_result(model)

        return result

    def get_result(self, model):
        if (len(model.getSols()) == 0) or (model.getStatus().lower() not in ['optimal', 'timelimit']):
            result = Result(
                infeasible=True,
                runtime=model.getSolvingTime(),
                objective=0,
                gap=-1,
                primal_curve=([0,], [0,]),
            )
        else:
            result = Result(
                infeasible=False,
                runtime=model.getSolvingTime(),
                objective=model.getObjVal(),
                gap=model.getGap(),
                primal_curve=model.get_primal_curve(),
            )

        return result

    def load_model(self, instance, hide_output=True) -> ModelWithPrimalDualIntegral:
        model = instance.to_scip(enable_primal_dual_integral=True,
                                 timeout=self.timeout)

        if hide_output:
            model.hideOutput()

        return model

class LearningBasedSolver(ABC,SCIPSolver):
    def __init__(self, net_wandb_id: str, n: int, timeout=60) -> None:
        super().__init__(timeout)

        self.n = n

        net_run = wandb.Api().run('brunompac/sat-gnn/'+net_wandb_id)
        net_config = net_run.config
        net_group = net_run.group
        net_file = wandb.restore('model_best.pth', run_path='/'.join(net_run.path),
                                 replace=True)

        try:
            self.net = OptSatGNN(
                n_h_feats=net_config['n_h_feats'],
                single_conv_for_both_passes=net_config['single_conv'],
                n_passes=net_config['n_passes'],
            )
            self.net.load_state_dict(torch.load(net_file.name))
        except RuntimeError:
            self.net = OptSatGNN(
                conv1='GraphConv', conv1_kwargs=dict(),
                n_h_feats=net_config['n_h_feats'],
                single_conv_for_both_passes=net_config['single_conv'],
                n_passes=net_config['n_passes'],
            )
            self.net.load_state_dict(torch.load(net_file.name))
        self.net.eval()

        self._net_config = net_config
        self._net_group = net_group

    def _get_prediction(self, instance: Instance):
        model = instance.to_gurobipy()
        graph = instance.to_graph(model=model)

        with torch.set_grad_enabled(False):
            x_hat = self.net.get_candidate(graph).flatten().cpu()
            x_hat = x_hat[:len(instance.vars_names)]  # drop zetas
        
        return x_hat
    
    def _get_candidate_from_prediction(self, instance: Instance, x_hat):
        most_certain_idx  = (x_hat - 0.5).abs().sort(descending=True).indices

        candidate_x_hat = (x_hat[most_certain_idx[:int(self.n)]] > .5).to(x_hat)
        candidate_vars_names = instance.vars_names[most_certain_idx[:int(self.n)]]
        candidate = dict(zip(candidate_vars_names, candidate_x_hat))

        return candidate

    def get_candidate_solution(self, instance: Instance):
        x_hat = self._get_prediction(instance)

        candidate = self._get_candidate_from_prediction(instance, x_hat)

        return candidate

class ConfLearningBasedSolver(LearningBasedSolver):
    def __init__(self, net_wandb_id: str, confidence_threshold: float, timeout=60) -> None:
        self.confidence_threshold = confidence_threshold
        super().__init__(net_wandb_id, None, timeout)

    @staticmethod
    def _get_pred_conf(x_hat):
        return (x_hat - 0.5).abs() + 0.5

    def _get_candidate_from_prediction(self, instance: Instance, x_hat):
        confidence = self._get_pred_conf(x_hat)

        candidate_x_hat = (x_hat[confidence > self.confidence_threshold] > .5).to(x_hat)
        candidate_vars_names = instance.vars_names[confidence > self.confidence_threshold]
        candidate = dict(zip(candidate_vars_names, candidate_x_hat))

        return candidate

class WarmStartingSolver(LearningBasedSolver):
    def load_model(self, instance, hide_output=True):
        model = super().load_model(instance, hide_output=hide_output)

        candidate = self.get_candidate_solution(instance)

        if candidate is not None:
            sol = model.createPartialSol()
            for var in model.getVars():
                try:
                    fixed_var_X = candidate[var.name]
                    # model_.fixVar(var, fixed_var_X)
                    model.setSolVal(sol, var, fixed_var_X)
                except KeyError:
                    pass
            model.addSol(sol)

        return model

class EarlyFixingSolver(LearningBasedSolver):
    def load_model(self, instance, hide_output=True):
        model = super().load_model(instance, hide_output=hide_output)

        candidate = self.get_candidate_solution(instance)

        if candidate is not None:
            for var in model.getVars():
                try:
                    fixed_var_X = candidate[var.name]
                    # model_.fixVar(var, fixed_var_X)
                    model.fixVar(var, fixed_var_X)
                except KeyError:
                    pass

        return model

class ConfEarlyFixingSolver(ConfLearningBasedSolver):
    def load_model(self, instance, hide_output=True):
        model = super().load_model(instance, hide_output=hide_output)

        candidate = self.get_candidate_solution(instance)

        if candidate is not None:
            for var in model.getVars():
                try:
                    fixed_var_X = candidate[var.name]
                    # model_.fixVar(var, fixed_var_X)
                    model.fixVar(var, fixed_var_X)
                except KeyError:
                    pass

        return model

class TrustRegionSolver(LearningBasedSolver):
    def __init__(self, net_wandb_id: str, n: int, Delta=1/20, timeout=60) -> None:
        super().__init__(net_wandb_id, n, timeout)

        if Delta < 1:
            self.Delta = int(n * Delta)
        else:
            self.Delta = int(Delta)

    def load_model(self, instance, hide_output=True):
        model = super().load_model(instance, hide_output=hide_output)

        candidate = self.get_candidate_solution(instance)

        if candidate is not None:
            abs_diffs = list()
            i = 0
            for var in model.getVars():
                if var.name in candidate.keys():
                    diff = var - candidate[var.name]

                    abs_diffs.append(model.addVar(name="diff(%s)" % i, lb=0, ub=1, vtype="CONTINUOUS"))
                    model.addCons(abs_diffs[-1] >= diff)
                    model.addCons(abs_diffs[-1] >= -diff)
            model.addCons(quicksum(abs_diffs) <= self.Delta)

        return model

class ConfidenceRegionSolver(ConfLearningBasedSolver):
    def __init__(self, net_wandb_id: str, k=1, timeout=60) -> None:
        # with confidence_threshold=-1, all variables are part of the candidate
        super().__init__(net_wandb_id, -1, timeout)

        self.k = float(k)

    def load_model(self, instance, hide_output=True):
        model = super().load_model(instance, hide_output=hide_output)

        x_hat = self._get_prediction(instance)
        candidate = self._get_candidate_from_prediction(instance, x_hat)

        confidence = self._get_pred_conf(x_hat)
        Delta = (1 - confidence).sum()

        if candidate is not None:
            abs_diffs = list()
            i = 0
            for var in model.getVars():
                if var.name in candidate.keys():
                    diff = var - candidate[var.name]

                    abs_diffs.append(model.addVar(name="diff(%s)" % i, lb=0, ub=1, vtype="CONTINUOUS"))
                    model.addCons(abs_diffs[-1] >= diff)
                    model.addCons(abs_diffs[-1] >= -diff)
            model.addCons(quicksum(abs_diffs) <= self.k * Delta)

        return model
