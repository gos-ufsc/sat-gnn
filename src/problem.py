from pathlib import Path
import numpy as np
import json
import pickle
import re
import torch
import gurobipy
from gurobipy import GRB

from pyscipopt import Model, quicksum


def random_instance(T, jobs, subs=1): 
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

    return {
        "subs": subs,
        "jobs": jobs,
        "T": T,

        "power_use": power_use.tolist(),
        "power_resource": power_resource.tolist(),

        "min_cpu_time": min_cpu_time.tolist(),
        "max_cpu_time": max_cpu_time.tolist(),
        "min_job_period": min_job_period.tolist(),
        "max_job_period": max_job_period.tolist(),
        "min_startup": min_startup.tolist(),
        "max_startup": max_startup.tolist(),
        "priority": priority.tolist(),
        "win_min": win_min.tolist(),
        "win_max": win_max.tolist(),
    }

def get_model(instance, coupling=True, recurso=None, new_ineq=False,
              timeout=60):
    if isinstance(instance, str) or isinstance(instance, Path):
        with open(instance) as f:
            instance = json.load(f)
    
    jobs = list(range(instance['jobs']))

    colunas_ = []
    lb = 0
    # T = instance['tamanho'][0]
    T = instance['T']

    if recurso is None:
        recurso_p = instance['power_resource']  # this was a typo while generating the instances
    else:
        recurso_p = recurso
    # print(recurso_p)

    priority = instance['priority'] # prioridade de cada tarefa
    uso_p = instance['power_use'] # recurso utilizado por cada tarefa
    min_statup = instance['min_startup'] # tempo mínimo de vezes que uma tarefa pode iniciar
    max_statup = instance['max_startup'] # tempo máximo de vezes que uma tarefa pode iniciar
    min_cpu_time = instance['min_cpu_time'] # tempo mínimo de unidades de tempo que uma tarefa pode consumir em sequência
    max_cpu_time = instance['max_cpu_time'] # tempo máximo de unidades de tempo que uma tarefa pode consumir em sequência
    min_periodo_job = instance['min_job_period'] # tempo mínimo que uma tarefa deve esperar para se repetir
    max_periodo_job = instance['max_job_period'] # tempo máximo que uma tarefa pode esperar para se repetir
    win_min = instance['win_min']
    win_max = instance['win_max']

    # create a model
    model = gurobipy.Model()
    model.Params.LogToConsole = 0

    if timeout is not None:
        model.setParam('TimeLimit', timeout)

    if isinstance(jobs, list):
        J_SUBSET = jobs
    else:
        J_SUBSET = [jobs]

    # create decision variables
    x = {}
    phi = {}
    for j in J_SUBSET:
        # the order in which we add the variables matter. I want all variables
        # associated with a given job to be together,
        # like x(0,0),...,x(0,T-1),phi(0,0),...phi(0,T-1),x(1,0),...,x(1,T-1),phi(1,0),...
        for t in range(T):
                x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1, vtype=GRB.BINARY)
        for t in range(T):
                phi[j,t] = model.addVar(vtype=GRB.BINARY, name="phi(%s,%s)" % (j, t),)

    soc_inicial = 0.7
    limite_inferior = 0.0
    ef = 0.9 
    v_bat = 3.6 
    q = 5
    bat_usage = 5

    # set objective
    model.setObjective(sum(priority[j] * x[j,t] for j in J_SUBSET for t in range(T)), GRB.MAXIMIZE)

    # phi defines startups of jobs
    for t in range(T):
        for j in J_SUBSET:
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
    for j in J_SUBSET:
        model.addConstr(sum(phi[j,t] for t in range(T)) >= min_statup[j], f"C6_j{j}")
        model.addConstr(sum(phi[j,t] for t in range(T)) <= max_statup[j], f"C7_j{j}")

        ###############################
        # precisa ajustar

        # execution window
        model.addConstr(sum(x[j,t] for t in range(win_min[j])) == 0, f"C8_j{j}")
        model.addConstr(sum(x[j,t] for t in range(win_max[j], T)) == 0, f"C9_j{j}")

    for j in J_SUBSET:
        # minimum period between jobs
        for t in range(0, T - min_periodo_job[j] + 1):
            model.addConstr(sum(phi[j,t_] for t_ in range(t, t + min_periodo_job[j])) <= 1, f"C10_j{j}_t{t}")

        # periodo máximo entre jobs
        for t in range(0, T - max_periodo_job[j] + 1):
            model.addConstr(sum(phi[j,t_] for t_ in range(t, t + max_periodo_job[j])) >= 1, f"C11_j{j}_t{t}")

        # min_cpu_time das jobs
        for t in range(0, T - min_cpu_time[j] + 1):
            model.addConstr(sum(x[j,t_] for t_ in range(t, t + min_cpu_time[j])) >= min_cpu_time[j] * phi[j,t], f"C12_j{j}_t{t}")

        # max_cpu_time das jobs
        for t in range(0, T - max_cpu_time[j]):
                model.addConstr(sum(x[j,t_] for t_ in range(t, t + max_cpu_time[j] + 1)) <= max_cpu_time[j], f"C13_j{j}_t{t}")

        # min_cpu_time no final do periodo
        for t in range(T - min_cpu_time[j] + 1, T):
                model.addConstr(sum(x[j,t_] for t_ in range(t, T)) >= (T - t) * phi[j,t], f"C14_j{j}_t{t}")

    if coupling:
        soc = {}
        i = {}
        b = {}
        for t in range(T):
            soc[t] = model.addVar(vtype=GRB.CONTINUOUS, name="soc(%s)" % t)
            i[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="i(%s)" % t)
            b[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b(%s)" % t)

        ################################
        # Add power constraints
        for t in range(T):
            model.addConstr(sum(uso_p[j] * x[j,t] for j in J_SUBSET) <= recurso_p[t] + bat_usage * v_bat, f"C15_t{t}")# * (1 - alpha[t]))

        ################################
        # Bateria
        ################################
        for t in range(T):
            model.addConstr(sum(uso_p[j] * x[j,t] for j in J_SUBSET) + b[t] == recurso_p[t], f"C16_t{t}")

        # Define the i_t, SoC_t constraints in Gurobi
        for t in range(T):
            # P = V * I 
            model.addConstr(b[t] / v_bat >= i[t], f"C17_t{t}")

            if t == 0:
                # SoC(1) = SoC(0) + p_carga[1]/60
                model.addConstr(soc[t] == soc_inicial + (ef / q) * (i[t] / 60), f"C18_t{t}")
            else:
                # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
                model.addConstr(soc[t] == soc[t - 1] + (ef / q) * (i[t] / 60), f"C19_t{t}")

            # Set the lower and upper limits on SoC
            model.addConstr(limite_inferior <= soc[t], f"C20_t{t}")
            model.addConstr(soc[t] <= 1, f"C21_t{t}")

    if new_ineq:
        # first
        for j in J_SUBSET:
            for t in range(T):
                model.addConstr(gurobipy.quicksum(phi[j, t_]
                                    for t_ in range(t, min(T, t + min_cpu_time[j] + 1))) <= 1,
                                                    name = f"VI_min_CPU_TIME_phi({j},{t})")

        # third
        for j in J_SUBSET:
            model.addConstr(gurobipy.quicksum(x[j, t] for t in range(T)) <=
                                    max_cpu_time[j]*gurobipy.quicksum(phi[j, t]
                                        for t in range(T)), name = f"VI_max_cpu_time_2({j})")
        # fourth
        for j in J_SUBSET:
            for t in range(0, T - max_cpu_time[j], 1):
                model.addConstr(
                    gurobipy.quicksum(x[j, t_] for t_ in range(t, t + max_cpu_time[j], 1)) <=
                        max_cpu_time[j]*gurobipy.quicksum(phi[j, t_] for t_ in range(max(t - max_cpu_time[j] + 1,0), t + max_cpu_time[j], 1)),
                    name = f"VI_max_cpu_time_3({j},{t})"
                )

        # fifth
        for j in J_SUBSET:
            for t in range(0, T - min_periodo_job[j] + 1):
                model.addConstr(gurobipy.quicksum(x[j, t_] for t_ in range(t, t + min_periodo_job[j])) <= min_periodo_job[j],
                                name = f"VI_min_period_btw_jobs_2({j},{t})")

        # sixth
        for j in J_SUBSET:
            if max_cpu_time[j] < (max_periodo_job[j] - min_cpu_time[j]):
                for t in range(0, T - max_cpu_time[j]):
                    model.addCons(phi[j,t] + x[j, t + max_cpu_time[j]] <= 1,
                                  name=f"VI_max_period_btw_jobs({j},{t})")

    model.update()

    return model

def get_model_scip(instance, coupling=True, recurso=None, new_ineq=False,
              timeout=60):
    if isinstance(instance, str) or isinstance(instance, Path):
        with open(instance) as f:
            instance = json.load(f)
    
    jobs = list(range(instance['jobs']))

    colunas_ = []
    lb = 0
    # T = instance['tamanho'][0]
    T = instance['T']

    if recurso is None:
        recurso_p = instance['power_resource']  # this was a typo while generating the instances
    else:
        recurso_p = recurso
    # print(recurso_p)

    priority = instance['priority'] # prioridade de cada tarefa
    uso_p = instance['power_use'] # recurso utilizado por cada tarefa
    min_statup = instance['min_startup'] # tempo mínimo de vezes que uma tarefa pode iniciar
    max_statup = instance['max_startup'] # tempo máximo de vezes que uma tarefa pode iniciar
    min_cpu_time = instance['min_cpu_time'] # tempo mínimo de unidades de tempo que uma tarefa pode consumir em sequência
    max_cpu_time = instance['max_cpu_time'] # tempo máximo de unidades de tempo que uma tarefa pode consumir em sequência
    min_periodo_job = instance['min_job_period'] # tempo mínimo que uma tarefa deve esperar para se repetir
    max_periodo_job = instance['max_job_period'] # tempo máximo que uma tarefa pode esperar para se repetir
    win_min = instance['win_min']
    win_max = instance['win_max']

    # create a model
    model = Model()
    # model.Params.LogToConsole = 0

    if timeout is not None:
        model.setParam('limits/time', timeout)

    if isinstance(jobs, list):
        J_SUBSET = jobs
    else:
        J_SUBSET = [jobs]

    # create decision variables
    x = {}
    phi = {}
    for j in J_SUBSET:
        # the order in which we add the variables matter. I want all variables
        # associated with a given job to be together,
        # like x(0,0),...,x(0,T-1),phi(0,0),...phi(0,T-1),x(1,0),...,x(1,T-1),phi(1,0),...
        for t in range(T):
                x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1, vtype="BINARY")
        for t in range(T):
                phi[j,t] = model.addVar(name="phi(%s,%s)" % (j, t), lb=0, ub=1, vtype="BINARY")

    soc_inicial = 0.7
    limite_inferior = 0.0
    ef = 0.9 
    v_bat = 3.6 
    q = 5
    bat_usage = 5

    # set objective
    model.setObjective(quicksum(priority[j] * x[j,t] for j in J_SUBSET for t in range(T)), "maximize")

    # phi defines startups of jobs
    for t in range(T):
        for j in J_SUBSET:
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
    for j in J_SUBSET:
        model.addCons(quicksum(phi[j,t] for t in range(T)) >= min_statup[j], f"C6_j{j}")
        model.addCons(quicksum(phi[j,t] for t in range(T)) <= max_statup[j], f"C7_j{j}")

        ###############################
        # precisa ajustar

        # execution window
        model.addCons(quicksum(x[j,t] for t in range(win_min[j])) == 0, f"C8_j{j}")
        model.addCons(quicksum(x[j,t] for t in range(win_max[j], T)) == 0, f"C9_j{j}")

    for j in J_SUBSET:
        # minimum period between jobs
        for t in range(0, T - min_periodo_job[j] + 1):
            model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + min_periodo_job[j])) <= 1, f"C10_j{j}_t{t}")

        # periodo máximo entre jobs
        for t in range(0, T - max_periodo_job[j] + 1):
            model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + max_periodo_job[j])) >= 1, f"C11_j{j}_t{t}")

        # min_cpu_time das jobs
        for t in range(0, T - min_cpu_time[j] + 1):
            model.addCons(quicksum(x[j,t_] for t_ in range(t, t + min_cpu_time[j])) >= min_cpu_time[j] * phi[j,t], f"C12_j{j}_t{t}")

        # max_cpu_time das jobs
        for t in range(0, T - max_cpu_time[j]):
                model.addCons(quicksum(x[j,t_] for t_ in range(t, t + max_cpu_time[j] + 1)) <= max_cpu_time[j], f"C13_j{j}_t{t}")

        # min_cpu_time no final do periodo
        for t in range(T - min_cpu_time[j] + 1, T):
                model.addCons(quicksum(x[j,t_] for t_ in range(t, T)) >= (T - t) * phi[j,t], f"C14_j{j}_t{t}")

    if coupling:
        soc = {}
        i = {}
        b = {}
        for t in range(T):
            soc[t] = model.addVar(vtype="CONTINUOUS", name="soc(%s)" % t)
            i[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="i(%s)" % t)
            b[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="b(%s)" % t)

        ################################
        # Add power constraints
        for t in range(T):
            model.addCons(quicksum(uso_p[j] * x[j,t] for j in J_SUBSET) <= recurso_p[t] + bat_usage * v_bat, f"C15_t{t}")# * (1 - alpha[t]))

        ################################
        # Bateria
        ################################
        for t in range(T):
            model.addCons(quicksum(uso_p[j] * x[j,t] for j in J_SUBSET) + b[t] == recurso_p[t], f"C16_t{t}")

        # Define the i_t, SoC_t constraints in Gurobi
        for t in range(T):
            # P = V * I 
            model.addCons(b[t] / v_bat >= i[t], f"C17_t{t}")

            if t == 0:
                # SoC(1) = SoC(0) + p_carga[1]/60
                model.addCons(soc[t] == soc_inicial + (ef / q) * (i[t] / 60), f"C18_t{t}")
            else:
                # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
                model.addCons(soc[t] == soc[t - 1] + (ef / q) * (i[t] / 60), f"C19_t{t}")

            # Set the lower and upper limits on SoC
            model.addCons(limite_inferior <= soc[t], f"C20_t{t}")
            model.addCons(soc[t] <= 1, f"C21_t{t}")

    if new_ineq:
        # first
        for j in J_SUBSET:
            for t in range(T):
                model.addCons(quicksum(phi[j, t_] for t_ in range(t, min(T, t + min_cpu_time[j] + 1))) <= 1,
                              name=f"VI_min_CPU_TIME_phi({j},{t})")

        # third
        for j in J_SUBSET:
            model.addCons(quicksum(x[j, t] for t in range(T)) <=
                                    max_cpu_time[j]*quicksum(phi[j, t]
                                        for t in range(T)), name=f"VI_max_cpu_time_2({j})")
        # fourth
        for j in J_SUBSET:
            for t in range(0, T - max_cpu_time[j], 1):
                model.addCons(
                    quicksum(x[j, t_] for t_ in range(t, t + max_cpu_time[j], 1)) <=
                        max_cpu_time[j]*quicksum(phi[j, t_] for t_ in range(max(t - max_cpu_time[j] + 1,0), t + max_cpu_time[j], 1)),
                    name=f"VI_max_cpu_time_3({j},{t})"
                )

        # fifth
        for j in J_SUBSET:
            for t in range(0, T - min_periodo_job[j] + 1):
                model.addCons(quicksum(x[j, t_] for t_ in range(t, t + min_periodo_job[j])) <= min_periodo_job[j],
                              name=f"VI_min_period_btw_jobs_2({j},{t})")

        # sixth
        for j in J_SUBSET:
            if max_cpu_time[j] < (max_periodo_job[j] - min_cpu_time[j]):
                for t in range(0, T - max_cpu_time[j]):
                    model.addCons(phi[j,t] + x[j, t + max_cpu_time[j]] <= 1,
                                  name=f"VI_max_period_btw_jobs({j},{t})")

    # model.update()

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

    bat = r - consumo
    i = bat / v_bat

    soc = torch.zeros_like(consumo).to(X)
    soc[:,0] = soc_inicial
    for t in range(1,T):
        soc[:,t] = soc[:,t-1] + (ef / q) * (i[:,t] / 60)

    return soc

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
