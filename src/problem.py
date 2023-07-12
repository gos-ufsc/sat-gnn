from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json
import dgl
import re
import torch
import gurobipy
from gurobipy import GRB

from pyscipopt import Model, quicksum, SCIP_EVENTTYPE
from pyscipopt.scip import Event, Eventhdlr


@dataclass
class Instance:
    """Instance of the ONTS problem."""
    jobs: int
    T: int
    power_use: list  # recurso utilizado por cada tarefa
    power_resource: list  # recurso disponível ao longo da trajetória
    min_cpu_time: list  # mínimo de unidades de tempo que uma tarefa pode consumir em sequência
    max_cpu_time: list  # máximo de unidades de tempo que uma tarefa pode consumir em sequência
    min_job_period: list  # tempo mínimo que uma tarefa deve esperar para se repetir
    max_job_period: list  # tempo máximo que uma tarefa pode esperar para se repetir
    min_startup: list  # tempo mínimo de vezes que uma tarefa pode iniciar
    max_startup: list  # tempo máximo de vezes que uma tarefa pode iniciar
    priority: list  # prioridade de cada tarefa
    win_min: list  # tamanho mínimo da janela
    win_max: list  # tamanho máximo da janela

    subs: int = 1
    initial_soc: float = 0.7
    lower_limit: float = 0.0
    ef: float = 0.9 
    v_bat: float = 3.6 
    q: float = 5
    bat_usage: float = 5

    @classmethod
    def from_file(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def random(cls, T, jobs, subs=1):
        orbit_power = np.loadtxt('/home/bruno/sat-gnn/data/raw/resource.csv')

        orbit_start = np.random.randint(0, 600)
        power_resource = orbit_power[orbit_start:orbit_start+T]

        min_power = 0.01
        max_power = 4.00
        power_use = np.random.rand(jobs)
        power_use = (max_power - min_power) * power_use + min_power

        priority = np.arange(jobs) + 1
        np.random.shuffle(priority)

        min_cpu_time = np.random.randint(1, T / 5, size=jobs)
        max_cpu_time = np.random.rand(jobs)
        max_cpu_time = max_cpu_time * (T / 2 - min_cpu_time) + min_cpu_time
        max_cpu_time = max_cpu_time.astype(int)

        max_possible_startup = (T / min_cpu_time).astype(int)
        min_startup = np.random.rand() * (max_possible_startup - 1) + 1
        min_startup = min_startup.astype(int)
        max_startup = np.random.rand(jobs)
        max_startup = max_startup * (T - min_startup) + min_startup
        max_startup = max_startup.astype(int)

        max_possible_job_period = (T / min_startup).astype(int)
        min_job_period = np.random.rand(jobs)
        min_job_period = min_job_period * (max_possible_job_period - 1) + 1
        min_job_period = min_job_period.astype(int)

        max_job_period = np.random.rand(jobs)
        max_job_period = max_job_period * (T - min_job_period) + min_job_period
        max_job_period = max_job_period.astype(int)

        win = np.eye(jobs)[0].astype(int)
        np.random.shuffle(win)
        win_min = win * np.random.randint(1, T * 1/5)
        win_max = win * np.random.randint(T * 4/5, T)
        win_max[win_max == 0] = T

        return cls(
            subs=subs,
            jobs=jobs,
            T=T,
            power_use=power_use.tolist(),
            power_resource=power_resource.tolist(),
            min_cpu_time=min_cpu_time.tolist(),
            max_cpu_time=max_cpu_time.tolist(),
            min_job_period=min_job_period.tolist(),
            max_job_period=max_job_period.tolist(),
            min_startup=min_startup.tolist(),
            max_startup=max_startup.tolist(),
            priority=priority.tolist(),
            win_min=win_min.tolist(),
            win_max=win_max.tolist(),
        )

    def to_gurobipy(self, coupling=True, new_inequalities=False, timeout=60) -> gurobipy.Model:
        # create blank model
        model = gurobipy.Model()
        model.Params.LogToConsole = 0

        if timeout is not None:
            model.setParam('TimeLimit', timeout)

        # create decision variables
        x = {}
        phi = {}
        for j in range(self.jobs):
            # the order in which we add the variables matters. I want all
            # variables associated with a given job to be together, like
            # x(0,0),...,x(0,T-1),phi(0,0),...phi(0,T-1),x(1,0),...,x(1,T-1),phi(1,0),...
            for t in range(self.T):
                    x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1,
                                          vtype=GRB.BINARY)
            for t in range(self.T):
                    phi[j,t] = model.addVar(name="phi(%s,%s)" % (j, t),
                                            vtype=GRB.BINARY)

        # set objective
        model.setObjective(sum(self.priority[j] * x[j,t]
                               for j in range(self.jobs)
                               for t in range(self.T)),
                           GRB.MAXIMIZE)

        # phi defines startups of jobs
        for t in range(self.T):
            for j in range(self.jobs):
                if t == 0:
                        model.addConstr(phi[j,t] >= x[j,t] - 0, f"C1_j{j}_t{t}")
                else:
                        model.addConstr(phi[j,t] >= x[j,t] - x[j,t - 1], f"C2_j{j}_t{t}")

                model.addConstr(phi[j,t] <= x[j,t], f"C3_j{j}_t{t}")

                if t == 0:
                        model.addConstr(phi[j,t] <= 2 - x[j,t] - 0, f"C4_j{j}_t{t}")
                else:
                        model.addConstr(phi[j,t] <= 2 - x[j,t] - x[j,t - 1], f"C5_j{j}_t{t}")

        # minimum and maximum number of startups of a job
        for j in range(self.jobs):
            model.addConstr(sum(phi[j,t] for t in range(self.T)) >= self.min_startup[j], f"C6_j{j}")
            model.addConstr(sum(phi[j,t] for t in range(self.T)) <= self.max_startup[j], f"C7_j{j}")

            ###############################
            # precisa ajustar

            # execution window
            model.addConstr(sum(x[j,t] for t in range(self.win_min[j])) == 0, f"C8_j{j}")
            model.addConstr(sum(x[j,t] for t in range(self.win_max[j], self.T)) == 0, f"C9_j{j}")

        for j in range(self.jobs):
            # minimum period between jobs
            for t in range(0, self.T - self.min_job_period[j] + 1):
                model.addConstr(sum(phi[j,t_] for t_ in range(t, t + self.min_job_period[j])) <= 1, f"C10_j{j}_t{t}")

            # periodo máximo entre jobs
            for t in range(0, self.T - self.max_job_period[j] + 1):
                model.addConstr(sum(phi[j,t_] for t_ in range(t, t + self.max_job_period[j])) >= 1, f"C11_j{j}_t{t}")

            # min_cpu_time das jobs
            for t in range(0, self.T - self.min_cpu_time[j] + 1):
                model.addConstr(sum(x[j,t_] for t_ in range(t, t + self.min_cpu_time[j])) >= self.min_cpu_time[j] * phi[j,t], f"C12_j{j}_t{t}")

            # max_cpu_time das jobs
            for t in range(0, self.T - self.max_cpu_time[j]):
                    model.addConstr(sum(x[j,t_] for t_ in range(t, t + self.max_cpu_time[j] + 1)) <= self.max_cpu_time[j], f"C13_j{j}_t{t}")

            # min_cpu_time no final do periodo
            for t in range(self.T - self.min_cpu_time[j] + 1, self.T):
                    model.addConstr(sum(x[j,t_] for t_ in range(t, self.T)) >= (self.T - t) * phi[j,t], f"C14_j{j}_t{t}")

        if coupling:
            soc = {}
            i = {}
            b = {}
            for t in range(self.T):
                soc[t] = model.addVar(vtype=GRB.CONTINUOUS, name="soc(%s)" % t)
                i[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="i(%s)" % t)
                b[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b(%s)" % t)

            ################################
            # Add power constraints
            for t in range(self.T):
                model.addConstr(sum(self.power_use[j] * x[j,t] for j in range(self.jobs)) <= self.power_resource[t] + self.bat_usage * self.v_bat, f"C15_t{t}")# * (1 - alpha[t]))

            ################################
            # Bateria
            ################################
            for t in range(self.T):
                model.addConstr(sum(self.power_use[j] * x[j,t] for j in range(self.jobs)) + b[t] == self.power_resource[t], f"C16_t{t}")

            # Define the i_t, SoC_t constraints in Gurobi
            for t in range(self.T):
                # P = V * I 
                model.addConstr(b[t] / self.v_bat >= i[t], f"C17_t{t}")

                if t == 0:
                    # SoC(1) = SoC(0) + p_carga[1]/60
                    model.addConstr(soc[t] == self.initial_soc + (self.ef / self.q) * (i[t] / 60), f"C18_t{t}")
                else:
                    # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
                    model.addConstr(soc[t] == soc[t - 1] + (self.ef / self.q) * (i[t] / 60), f"C19_t{t}")

                # Set the lower and upper limits on SoC
                model.addConstr(self.lower_limit <= soc[t], f"C20_t{t}")
                model.addConstr(soc[t] <= 1, f"C21_t{t}")

        if new_inequalities:
            # first
            for j in range(self.jobs):
                for t in range(self.T):
                    model.addConstr(
                        gurobipy.quicksum(
                                phi[j, t_] for t_ in range(
                                    t, min(self.T, t + self.min_cpu_time[j] + 1)
                                )
                            ) <= 1,
                        name = f"VI_min_CPU_TIME_phi({j},{t})"
                    )

            # third
            for j in range(self.jobs):
                model.addConstr(
                    gurobipy.quicksum(x[j, t] for t in range(self.T)) <=
                        self.max_cpu_time[j] * gurobipy.quicksum(
                            phi[j, t] for t in range(self.T)
                        ),
                    name = f"VI_max_cpu_time_2({j})"
                )

            # fourth
            for j in range(self.jobs):
                for t in range(0, self.T - self.max_cpu_time[j], 1):
                    model.addConstr(
                        gurobipy.quicksum(x[j, t_] for t_ in range(
                                t, t + self.max_cpu_time[j], 1
                            )) <= self.max_cpu_time[j] * gurobipy.quicksum(
                                phi[j, t_] for t_ in range(
                                    max(t - self.max_cpu_time[j] + 1,0),
                                    t + self.max_cpu_time[j],
                                    1,
                                )
                            ),
                        name = f"VI_max_cpu_time_3({j},{t})"
                    )

            # fifth
            for j in range(self.jobs):
                for t in range(0, self.T - self.min_job_period[j] + 1):
                    model.addConstr(gurobipy.quicksum(x[j, t_] for t_ in range(t, t + self.min_job_period[j])) <= self.min_job_period[j],
                                    name = f"VI_min_period_btw_jobs_2({j},{t})")

            # sixth
            for j in range(self.jobs):
                if self.max_cpu_time[j] < (self.max_job_period[j] - self.min_cpu_time[j]):
                    for t in range(0, self.T - self.max_cpu_time[j]):
                        model.addConstr(phi[j,t] + x[j, t + self.max_cpu_time[j]] <= 1,
                                    name=f"VI_max_period_btw_jobs({j},{t})")

        model.update()

        return model

    def to_scip(self, coupling=True, new_inequalities=False, timeout=60,
                enable_primal_dual_integral=True) -> Model:
        # create a model
        if enable_primal_dual_integral:
            model = ModelWithPrimalDualIntegral()
        else:
            model = Model()

        if timeout is not None:
            model.setParam('limits/time', timeout)

        # create decision variables
        x = {}
        phi = {}
        for j in range(self.jobs):
            # the order in which we add the variables matter. I want all
            # variables associated with a given job to be together, like
            # x(0,0),...,x(0,T-1),phi(0,0),...phi(0,T-1),x(1,0),...,x(1,T-1),phi(1,0),...
            for t in range(self.T):
                    x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1,
                                          vtype="BINARY")
            for t in range(self.T):
                    phi[j,t] = model.addVar(name="phi(%s,%s)" % (j, t), lb=0, ub=1,
                                            vtype="BINARY")

        model.setObjective(quicksum(self.priority[j] * x[j,t]
                                    for j in range(self.jobs)
                                    for t in range(self.T)),
                           "maximize")

        # phi defines startups of jobs
        for t in range(self.T):
            for j in range(self.jobs):
                if t == 0:
                        model.addCons(phi[j,t] >= x[j,t] - 0, f"C1_j{j}_t{t}")
                else:
                        model.addCons(phi[j,t] >= x[j,t] - x[j,t - 1], f"C2_j{j}_t{t}")

                model.addCons(phi[j,t] <= x[j,t], f"C3_j{j}_t{t}")

                if t == 0:
                        model.addCons(phi[j,t] <= 2 - x[j,t] - 0, f"C4_j{j}_t{t}")
                else:
                        model.addCons(phi[j,t] <= 2 - x[j,t] - x[j,t - 1], f"C5_j{j}_t{t}")

        # minimum and maximum number of startups of a job
        for j in range(self.jobs):
            model.addCons(quicksum(phi[j,t] for t in range(self.T)) >= self.min_startup[j], f"C6_j{j}")
            model.addCons(quicksum(phi[j,t] for t in range(self.T)) <= self.max_startup[j], f"C7_j{j}")

            ###############################
            # precisa ajustar

            # execution window
            model.addCons(quicksum(x[j,t] for t in range(self.win_min[j])) == 0, f"C8_j{j}")
            model.addCons(quicksum(x[j,t] for t in range(self.win_max[j], self.T)) == 0, f"C9_j{j}")

        for j in range(self.jobs):
            # minimum period between jobs
            for t in range(0, self.T - self.min_job_period[j] + 1):
                model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + self.min_job_period[j])) <= 1, f"C10_j{j}_t{t}")

            # periodo máximo entre jobs
            for t in range(0, self.T - self.max_job_period[j] + 1):
                model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + self.max_job_period[j])) >= 1, f"C11_j{j}_t{t}")

            # min_cpu_time das jobs
            for t in range(0, self.T - self.min_cpu_time[j] + 1):
                model.addCons(quicksum(x[j,t_] for t_ in range(t, t + self.min_cpu_time[j])) >= self.min_cpu_time[j] * phi[j,t], f"C12_j{j}_t{t}")

            # max_cpu_time das jobs
            for t in range(0, self.T - self.max_cpu_time[j]):
                    model.addCons(quicksum(x[j,t_] for t_ in range(t, t + self.max_cpu_time[j] + 1)) <= self.max_cpu_time[j], f"C13_j{j}_t{t}")

            # min_cpu_time no final do periodo
            for t in range(self.T - self.min_cpu_time[j] + 1, self.T):
                    model.addCons(quicksum(x[j,t_] for t_ in range(t, self.T)) >= (self.T - t) * phi[j,t], f"C14_j{j}_t{t}")

        if coupling:
            soc = {}
            i = {}
            b = {}
            for t in range(self.T):
                soc[t] = model.addVar(vtype="CONTINUOUS", name="soc(%s)" % t)
                i[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="i(%s)" % t)
                b[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="b(%s)" % t)

            ################################
            # Add power constraints
            for t in range(self.T):
                model.addCons(quicksum(self.power_use[j] * x[j,t] for j in range(self.jobs)) <= self.power_resource[t] + self.bat_usage * self.v_bat, f"C15_t{t}")# * (1 - alpha[t]))

            ################################
            # Bateria
            ################################
            for t in range(self.T):
                model.addCons(quicksum(self.power_use[j] * x[j,t] for j in range(self.jobs)) + b[t] == self.power_resource[t], f"C16_t{t}")

            # Define the i_t, SoC_t constraints in Gurobi
            for t in range(self.T):
                # P = V * I 
                model.addCons(b[t] / self.v_bat >= i[t], f"C17_t{t}")

                if t == 0:
                    # SoC(1) = SoC(0) + p_carga[1]/60
                    model.addCons(soc[t] == self.initial_soc + (self.ef / self.q) * (i[t] / 60), f"C18_t{t}")
                else:
                    # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
                    model.addCons(soc[t] == soc[t - 1] + (self.ef / self.q) * (i[t] / 60), f"C19_t{t}")

                # Set the lower and upper limits on SoC
                model.addCons(self.lower_limit <= soc[t], f"C20_t{t}")
                model.addCons(soc[t] <= 1, f"C21_t{t}")

        if new_inequalities:
            # first
            for j in range(self.jobs):
                for t in range(self.T):
                    model.addCons(
                        quicksum(phi[j, t_] for t_ in range(
                            t,
                            min(self.T, t + self.min_cpu_time[j] + 1)
                        )) <= 1,
                        name=f"VI_min_CPU_TIME_phi({j},{t})"
                    )

            # third
            for j in range(self.jobs):
                model.addCons(
                    quicksum(x[j, t] for t in range(self.T)) <=
                        self.max_cpu_time[j] * quicksum(
                            phi[j, t] for t in range(self.T)
                        ),
                    name = f"VI_max_cpu_time_2({j})"
                )

            # fourth
            for j in range(self.jobs):
                for t in range(0, self.T - self.max_cpu_time[j], 1):
                    model.addCons(
                        quicksum(x[j, t_] for t_ in range(
                                t, t + self.max_cpu_time[j], 1
                            )) <= self.max_cpu_time[j] * quicksum(
                                phi[j, t_] for t_ in range(
                                    max(t - self.max_cpu_time[j] + 1,0),
                                    t + self.max_cpu_time[j],
                                    1,
                                )
                            ),
                        name = f"VI_max_cpu_time_3({j},{t})"
                    )

            # fifth
            for j in range(self.jobs):
                for t in range(0, self.T - self.min_job_period[j] + 1):
                    model.addCons(quicksum(x[j, t_] for t_ in range(t, t + self.min_job_period[j])) <= self.min_job_period[j],
                                    name = f"VI_min_period_btw_jobs_2({j},{t})")

            # sixth
            for j in range(self.jobs):
                if self.max_cpu_time[j] < (self.max_job_period[j] - self.min_cpu_time[j]):
                    for t in range(0, self.T - self.max_cpu_time[j]):
                        model.addCons(phi[j,t] + x[j, t + self.max_cpu_time[j]] <= 1,
                                    name=f"VI_max_period_btw_jobs({j},{t})")

        return model

    def to_graph(self, model: gurobipy.Model = None, **model_kwargs) -> dgl.DGLHeteroGraph:
        if model is None:
            model = self.to_gurobipy(**model_kwargs)

        # TODO: include variable bounds (not present in getA())
        A = model.getA().toarray()
        # TODO: include sos variable constraints
        b = np.array(model.getAttr('rhs'))
        c = np.array(model.getAttr('obj'))

        # get only real (non-null) edges
        A_ = A.flatten()
        edges = np.indices(A.shape)  # cons -> vars
        edges = edges.reshape(edges.shape[0],-1)
        edges = edges[:,A_ != 0]
        # edges = torch.from_numpy(edges)

        edge_weights = A_[A_ != 0]

        constraints_sense = np.array([ci.sense for ci in model.getConstrs()])
        constraints_sense = np.array(list(map({'>': 1, '=': 0, '<': -1}.__getitem__, constraints_sense)))

        vars_names = [v.getAttr(GRB.Attr.VarName) for v in model.getVars()]
        # grab all non-decision variables (everything that is not `x` or `phi`)
        soc_vars_mask = np.array([('x(' not in v) and ('phi(' not in v) and ('zeta(' not in v) for v in vars_names])
        soc_vars = np.arange(soc_vars_mask.shape[0])[soc_vars_mask]
        var_vars = np.arange(soc_vars_mask.shape[0])[~soc_vars_mask]
        soc_edges_mask = np.isin(edges.T[:,1], soc_vars)

        var_edges = edges[:,~soc_edges_mask]
        soc_edges = edges[:,soc_edges_mask]

        # translate soc/var nodes index to 0-based
        soc_edges[1] = np.array(list(map(
            dict(zip(soc_vars, np.arange(soc_vars.shape[0]))).get,
            soc_edges[1]
        )))
        var_edges[1] = np.array(list(map(
            dict(zip(var_vars, np.arange(var_vars.shape[0]))).get,
            var_edges[1]
        )))

        g = dgl.heterograph({
            ('var', 'v2c', 'con'): (var_edges[1], var_edges[0]),
            ('con', 'c2v', 'var'): (var_edges[0], var_edges[1]),
            ('soc', 's2c', 'con'): (soc_edges[1], soc_edges[0]),
            ('con', 'c2s', 'soc'): (soc_edges[0], soc_edges[1]),
        })

        soc_edge_weights = edge_weights[soc_edges_mask]
        g.edges['s2c'].data['A'] = torch.from_numpy(soc_edge_weights)
        g.edges['c2s'].data['A'] = torch.from_numpy(soc_edge_weights)

        var_edge_weights = edge_weights[~soc_edges_mask]
        g.edges['v2c'].data['A'] = torch.from_numpy(var_edge_weights)
        g.edges['c2v'].data['A'] = torch.from_numpy(var_edge_weights)

        g.nodes['con'].data['x'] = torch.from_numpy(np.stack((
            b,  # rhs
            A.mean(1),  # c_coeff
            g.in_degrees(etype='v2c').numpy() + \
            g.in_degrees(etype='s2c').numpy(),  # Nc_coeff
            constraints_sense,  # sense
        ), -1))

        g.nodes['var'].data['x'] = torch.from_numpy(np.stack((
            c[~soc_vars_mask],  # obj
            A.mean(0)[~soc_vars_mask],  # v_coeff
            g.in_degrees(etype='c2v').numpy(),  # Nv_coeff
            A.max(0)[~soc_vars_mask],  # max_coeff
            A.min(0)[~soc_vars_mask],  # min_coeff
            np.ones_like(c[~soc_vars_mask]),  # int
            np.array([float(v.rstrip(')').split(',')[-1]) / 97 for v in np.array(vars_names)[~soc_vars_mask]]),  # pos_emb (kind of)
        ), -1))

        g.nodes['soc'].data['x'] = torch.from_numpy(np.stack((
            c[soc_vars_mask],  # obj
            A.mean(0)[soc_vars_mask],  # v_coeff
            g.in_degrees(etype='c2s').numpy(),  # Nv_coeff
            A.max(0)[soc_vars_mask],  # max_coeff
            A.min(0)[soc_vars_mask],  # min_coeff
            np.zeros_like(c[soc_vars_mask]),  # int
        ), -1))

        return g

class PrimalDualIntegralHandler(Eventhdlr):
    def __init__(self, eventtypes=[SCIP_EVENTTYPE.NODESOLVED, SCIP_EVENTTYPE.BESTSOLFOUND],
                 initial_primal=0, initial_dual=1e5):
        """`initial_primal/dual` punishes for longer times before first feasible.
        """
        self.eventtypes = eventtypes

        self._eventtype = 0
        for eventtype in self.eventtypes:
            self._eventtype |= eventtype

        self.primals = [initial_primal]
        self.duals = [initial_dual]
        self.times = [0]

        self.calls = list()

    def eventinit(self):
        self.model.catchEvent(self._eventtype, self)

    def eventexit(self):
        self.model.dropEvent(self._eventtype, self)

    def eventexec(self, event: Event):
        try:
            self.times.append(self.model.getTotalTime())
            primal = self.model.getPrimalbound()
            if primal < self.primals[0]:
                primal = self.primals[0]
            self.primals.append(primal)
            self.duals.append(self.model.getDualbound())
        except Exception:
            pass

    def get_primal_dual_integral(self):
        sense = 1 if self.model.getObjectiveSense() == 'minimze' else -1
        last_time = self.model.getTotalTime()

        times_ = self.times + [last_time,]
        integral = 0
        for i in range(len(times_) - 1):
            dt = times_[i+1] - times_[i]
            gap = sense * (self.primals[i] - self.duals[i])

            integral += dt * gap

        return integral

    def get_relative_primal_integral(self, reference: float):
        # sense = 1 if self.model.getObjectiveSense() == 'minimze' else -1
        last_time = self.model.getTotalTime()

        times_ = self.times + [last_time,]
        integral = 0
        for i in range(len(times_) - 1):
            dt = times_[i+1] - times_[i]
            relative_primal = 1 - self.primals[i-1] / reference

            integral += dt * relative_primal

        return integral

class ModelWithPrimalDualIntegral(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.primal_dual_handler = PrimalDualIntegralHandler()
        self.includeEventhdlr(self.primal_dual_handler, "Primal Dual Handler",
                              "Catches changes in the primal and the dual bounds")

    def get_primal_dual_integral(self, *args, **kwargs):
        return self.primal_dual_handler.get_primal_dual_integral(*args, **kwargs)
    
    def get_relative_primal_integral(self, *args, **kwargs):
        return self.primal_dual_handler.get_relative_primal_integral(*args, **kwargs)
    
    def get_primal_curve(self):
        return self.primal_dual_handler.primals, self.primal_dual_handler.times

def add_trust_region(model, values: dict(), Delta=15):
    abs_diffs = list()
    i = 0
    for var in model.getVars():
        if var.name in values.keys():
            diff = var - values[var.name]

            abs_diffs.append(model.addVar(name="diff(%s)" % i, lb=0, ub=1, vtype="CONTINUOUS"))
            model.addCons(abs_diffs[-1] >= diff)
            model.addCons(abs_diffs[-1] >= -diff)
    model.addCons(quicksum(abs_diffs) <= Delta)

    return model

def get_coupling_constraints(X, instance, r=None):
    J = instance['jobs'][0]
    T = instance['tamanho'][0]
    uso_p = torch.Tensor(instance['uso_p']).to(X) # recurso utilizado por cada tarefa

    if r is None:
        r = torch.Tensor(instance['recurso_p'])[:T]

    # parameters
    soc_inicial = 0.7
    limite_inferior = 0.0
    ef = 0.9
    v_bat = 3.6
    q = 5
    bat_usage = 5

    # format candidate solution for each job
    n_batch = X.shape[0]
    X = X.view(n_batch, J, 2 * T)
    X = X[:,:,:T]  # discard phi variables

    consumo = torch.bmm(uso_p.unsqueeze(0).repeat(n_batch,1,1), X).squeeze(1)
    recurso_total = r + bat_usage * v_bat

    bat = r - consumo
    i = bat / v_bat

    soc = torch.zeros_like(consumo).to(X)
    soc[:,0] = soc_inicial
    for t in range(1,T):
        soc[:,t] = soc[:,t-1] + (ef / q) * (i[:,t] / 60)

    g = torch.hstack((recurso_total - consumo, 1 - soc, soc - limite_inferior))

    return g

def get_soc(X, instance, r=None):
    J = instance['jobs']
    T = instance['T']
    uso_p = torch.Tensor(instance['power_use']).to(X) # recurso utilizado por cada tarefa

    if r is None:
        r = torch.Tensor(instance['power_resource'])[:T]

    # parameters
    soc_inicial = 0.7
    limite_inferior = 0.0
    ef = 0.9
    v_bat = 3.6
    q = 5
    bat_usage = 5

    # format candidate solution for each job
    n_batch = X.shape[0]
    X = X.view(n_batch, J, 2 * T)
    X = X[:,:,:T]  # discard phi variables

    consumo = torch.bmm(uso_p.unsqueeze(0).repeat(n_batch,1,1), X).squeeze(1)

    bat = r - consumo
    i = bat / v_bat

    soc = torch.zeros_like(consumo).to(X)
    soc[:,0] = soc_inicial
    for t in range(1,T):
        soc[:,t] = soc[:,t-1] + (ef / q) * (i[:,t] / 60)

    return torch.stack((soc, i, bat), dim=-1).flatten(1)

def get_benders_cut(instance, solucao, verbose=False):
    subproblem = Model()
    # if verbose:
        # subproblem.Params.LogToConsole = 1

    lmbd0 = {}
    lmbd1 = {}
    lmbd2 = {}
    lmbd3 = {}
    lmbd4 = {}
    lmbd5 = {}
    lmbd6 = {}
    lmbd7 = {}

    J = instance['jobs'][0]
    T = instance['tamanho'][0]
    uso_p = instance['uso_p']
    recurso_p = instance['recurso_p']

    soc_inicial = 0.7
    limite_inferior = 0.0
    ef = 0.9
    v_bat = 3.6
    q = 5
    bat_usage = 5

    for t in range(T):
        lmbd1[t] = subproblem.addVar(name="lmbd1(%s)" % t, lb=0, vtype=GRB.CONTINUOUS) # alpha 1 
        lmbd2[t] = subproblem.addVar(name="lmbd2(%s)" % t, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # b 2 
        lmbd3[t] = subproblem.addVar(name="lmbd3(%s)" % t, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # b 3
        lmbd4[t] = subproblem.addVar(name="lmbd4(%s)" % t, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # soc - i 4
        lmbd5[t] = subproblem.addVar(name="lmbd5(%s)" % t, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # soc - i 5
        lmbd6[t] = subproblem.addVar(name="lmbd6(%s)" % t, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # limite inferior 6
        lmbd7[t] = subproblem.addVar(name="lmbd7(%s)" % t, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS) # limite inferior 6

    for t in range(T):
        subproblem.addConstr((bat_usage * v_bat)*lmbd1[t] + lmbd7[t] >= 0)
        subproblem.addConstr(lmbd2[t] - lmbd3[t]/v_bat == 0)
        subproblem.addConstr(lmbd3[t] -  (ef / q)*(1/60) * lmbd4[t] == 0)

    for t in range(T-1):
            subproblem.addConstr(lmbd4[t] - lmbd4[t+1] + (-lmbd5[t] + lmbd6[t]) == 0)

    subproblem.addConstr(lmbd4[T-1] + (-lmbd5[T-1] + lmbd6[T-1]) == 0)
    subproblem.setParam('InfUnbdInfo', 1)
    # subproblem.Params.LogToConsole = 0
    subproblem.update()

    obj = 0

    for t in range(T):
        lhs = float(sum(uso_p[j] * solucao[j][t] for j in range(J)))
        obj += lmbd1[t] * ((recurso_p[t] + bat_usage * v_bat) - lhs) + lmbd2[t] * (recurso_p[t] - lhs) - 0.0*lmbd5[t] + lmbd6[t] + lmbd7[t]
    obj += lmbd4[0] * 0.7
    subproblem.setObjective(obj, GRB.MINIMIZE)

    subproblem.update()
    subproblem.optimize()

    if subproblem.status == 5:
        teste = gurobipy.Model()
        A = teste.addVars(J,T, vtype=GRB.BINARY, name="A")
        teste.update()
        cut = 0
        for t in range(T):
            lhs = sum(uso_p[j] * A[j,t] for j in range(J))
            cut += (lmbd1[t].getAttr(GRB.Attr.UnbdRay) * ((recurso_p[t] + bat_usage * v_bat) - lhs) + lmbd2[t].getAttr(GRB.Attr.UnbdRay) * (recurso_p[t] - lhs)
                    - lmbd5[t].getAttr(GRB.Attr.UnbdRay)*0.0 + lmbd6[t].getAttr(GRB.Attr.UnbdRay) + lmbd7[t].getAttr(GRB.Attr.UnbdRay) )
        cut += lmbd4[0].getAttr(GRB.Attr.UnbdRay) * 0.7
        cut = str(cut)

        indices = {}
        corte = []
        # Extracting the indices and coefficients using regular expression
        for match in re.finditer(r"([+-]?\d+\.\d+) A\[(\d+),(\d+)\]", cut):
            coefficient = float(match.group(1))
            i = int(match.group(2))
            j = int(match.group(3))
            indices[(i,j)] = float(coefficient)
            #if float(coefficient) != 0:
            #    print(i,j,coefficient)
            #    #print(pato)

        # assuming the cut is of the form w^T x >= b
        cut_w = np.zeros_like(solucao, dtype=float)
        for (i, j), w_ij in indices.items():
            cut_w[i,j] = w_ij
        cut_b = float(cut.split(' ')[0])
        # indices['const'] = cut.split(' ')[0]
        # #corte = [val for val in indices.values()]
        # for j in range(J):
        #     for t in range(T):
        #         corte.append(indices[(j,t)])
        # corte.append(indices['const'])

        return cut_w, cut_b
    elif subproblem.status == 2:
        return None
    else:
        print('ERROR: status ', subproblem.status)

def get_feasible(model, incumbent, instance, with_phi=True, weighted=False):
    model_ = model.copy()

    J = instance['jobs'][0]
    T = instance['tamanho'][0]

    if with_phi:
        T_ = T * 2
    else:
        T_ = T

    expr = 0
    for j in range(J):
        for t_ in range(T_):
            if t_ >= T:
                t = t_ - T
                var = 'phi'
            else:
                t = t_
                var = 'x'
            #if model.getVarByName("x(%s,%s,%s)" % (s,j,t)).x > 0.5:
            if incumbent[j][t] > 0.5:
                if weighted:
                    weight = 2 * (incumbent[j][t] - 0.5)
                else:
                    weight = 1
                expr += (1 - model_.getVarByName("%s(%s,%s)" % (var,j,t))) * weight
            else:
                if weighted:
                    weight = 2 * (0.5 - incumbent[j][t])
                else:
                    weight = 1
                expr += model_.getVarByName("%s(%s,%s)" % (var,j,t))
            #if incumbent_phi[j][t] > 0.5:
            #    expr += (1 - model.getVarByName("phi(%s,%s)" % (j,t)))
            #else:
            #    expr += model.getVarByName("phi[%s,%s,%s]" % (j,t))

    theta = model_.addVar(vtype=GRB.CONTINUOUS, name="theta")
    model_.addConstr(expr <= theta)
    M1 = 1000
    model_.setObjective(M1*theta, GRB.MINIMIZE)
    model_.update()
    model_.optimize()

    if model_.status == 2:
        feas_incumbent = np.zeros_like(incumbent)
        for var in model_.getVars():
            try:
                j, t = re.fullmatch(r"x\((\d+),(\d+)\)", var.getAttr(GRB.Attr.VarName)).groups()
            except AttributeError:
                continue
            j = int(j)
            t = int(t)
            feas_incumbent[j,t] = var.X

        return feas_incumbent

def get_vars_from_x(x, model):
    model_ = model.copy()

    for j in range(x.shape[0]):
        for t in range(x.shape[1]):
            model_.getVarByName(f"x({j},{t})").lb = x[j,t]
            model_.getVarByName(f"x({j},{t})").ub = x[j,t]

    model_.update()
    model_.optimize()

    return model_.getVars()
