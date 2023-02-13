from pathlib import Path
import numpy as np
import pickle
import re
import torch
import gurobipy


def load_instance(fpath="data/raw/97_9.jl"):
    instancia = {}
    interesses = ["jobs", "recurso_p", "tamanho", "priority", "uso_p",
                  "min_statup", "max_statup", "min_cpu_time", "max_cpu_time",
                  "min_periodo_job", "max_periodo_job", "win_min", "win_max"]

    for interesse in interesses:
        with open(fpath, "r") as exemplo:
            lines = exemplo.readlines()
            for line in lines:
                # check if string present on a current line
                if line.find(interesse) != -1:
                    if interesse == "uso_p" or interesse=="recurso_p":
                        dados = re.findall(r'\d+\.\d+',line)
                        instancia[interesse] = [float(dado) for dado in dados]
                    else:
                        dados = re.findall(r'\d+',line)
                        instancia[interesse] = [int(dado) for dado in dados]

    return instancia

def oracle(jobs, instance):
    if isinstance(instance, str) or isinstance(instance, Path):
        instance = load_instance(instance)

    colunas_ = []
    lb = 0
    T = instance['tamanho'][0]
    recurso_p = instance['recurso_p']
    # print(recurso_p)

    priority = instance['priority'] # prioridade de cada tarefa
    uso_p = instance['uso_p'] # recurso utilizado por cada tarefa
    min_statup = instance['min_statup'] # tempo mínimo de vezes que uma tarefa pode iniciar
    max_statup = instance['max_statup'] # tempo máximo de vezes que uma tarefa pode iniciar
    min_cpu_time = instance['min_cpu_time'] # tempo mínimo de unidades de tempo que uma tarefa pode consumir em sequência
    max_cpu_time = instance['max_cpu_time'] # tempo máximo de unidades de tempo que uma tarefa pode consumir em sequência
    min_periodo_job = instance['min_periodo_job'] # tempo mínimo que uma tarefa deve esperar para se repetir
    max_periodo_job = instance['max_periodo_job'] # tempo máximo que uma tarefa pode esperar para se repetir
    win_min = instance['win_min']
    win_max = instance['win_max']

    # create a model
    model = gurobipy.Model()
    model.Params.LogToConsole = 0
    model.setParam('PoolSearchMode', 1)
    model.setParam('PoolSolutions', 100000)
    model.setParam('TimeLimit', 120)
    # create decision variables

    if isinstance(jobs, list):
        J_SUBSET = jobs
    else:
        J_SUBSET = [jobs]

    x = {}
    phi = {}
    for j in J_SUBSET:
        # the order in which we add the variables matter. I want all variables
        # associated with a given job to be together,
        # like x(0,0),...,x(0,-1),phi(0,0),...phi(0,-1),x(1,0),...,x(1,-1),phi(1,0),...
        for t in range(T):
                x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1, vtype=gurobipy.GRB.BINARY)
        for t in range(T):
                phi[j,t] = model.addVar(vtype=gurobipy.GRB.BINARY, name="phi(%s,%s)" % (j, t),)

    # soc_inicial = 0.7
    # limite_inferior = 0.0
    # ef = 0.9 
    # v_bat = 3.6 
    # q = 5
    # bat_usage = 5

    # set objective
    model.setObjective(sum(priority[j] * x[j,t] for j in J_SUBSET for t in range(T)), gurobipy.GRB.MAXIMIZE)

    # phi defines startups of jobs
    for t in range(T):
        for j in J_SUBSET:
            if t == 0:
                    model.addConstr(phi[j,t] >= x[j,t] - 0)
            else:
                    model.addConstr(phi[j,t] >= x[j,t] - x[j,t - 1])

            model.addConstr(phi[j,t] <= x[j,t])

            if t == 0:
                    model.addConstr(phi[j,t] <= 2 - x[j,t] - 0)
            else:
                    model.addConstr(phi[j,t] <= 2 - x[j,t] - x[j,t - 1])

    # minimum and maximum number of startups of a job
    for j in J_SUBSET:
        model.addConstr(sum(phi[j,t] for t in range(T)) >= min_statup[j])
        model.addConstr(sum(phi[j,t] for t in range(T)) <= max_statup[j])

        ###############################
        # precisa ajustar

        # execution window
        model.addConstr(sum(x[j,t] for t in range(win_min[j])) == 0)
        model.addConstr(sum(x[j,t] for t in range(win_max[j], T)) == 0)

    for j in J_SUBSET:
        # minimum period between jobs
        for t in range(0, T - min_periodo_job[j] + 1):
            model.addConstr(sum(phi[j,t_] for t_ in range(t, t + min_periodo_job[j])) <= 1)

        # periodo máximo entre jobs
        for t in range(0, T - max_periodo_job[j] + 1):
            model.addConstr(sum(phi[j,t_] for t_ in range(t, t + max_periodo_job[j])) >= 1)

        # min_cpu_time das jobs
        for t in range(0, T - min_cpu_time[j] + 1):
            model.addConstr(sum(x[j,t_] for t_ in range(t, t + min_cpu_time[j])) >= min_cpu_time[j] * phi[j,t])

        # max_cpu_time das jobs
        for t in range(0, T - max_cpu_time[j]):
                model.addConstr(sum(x[j,t_] for t_ in range(t, t + max_cpu_time[j] + 1)) <= max_cpu_time[j])

        # min_cpu_time no final do periodo
        for t in range(T - min_cpu_time[j] + 1, T):
                model.addConstr(sum(x[j,t_] for t_ in range(t, T)) >= (T - t) * phi[j,t])

    model.update()

    return model

def load_data(instance):
    if isinstance(instance, str) or isinstance(instance, Path):
        instance = load_instance(instance)

    T = instance['tamanho'][0]
    J = instance['jobs'][0]
    JOBS = J

    A = []
    resultados = [[] for i in range(JOBS)]
    b = []
    c = []
    objetivos = []
    for job in range(JOBS):
        model = oracle(job, instance)
        #print(resultados, objetivos)
        A.append(model.getA().toarray())
        #resultados.append(np.array(resultados))
        b.append(np.array(model.getAttr('rhs')))
        c.append(np.array(model.getAttr('obj')))

        A[job].shape, b[job].shape, c[job].shape
        # TODO: this absolute path here is flimsy, I should do sth about it
        with open('/home/bruno/sat-gnn/data/processed/resultados_'+str(job)+'.pkl', 'rb') as f:
            resultadosX = pickle.load(f)
        with open('/home/bruno/sat-gnn/data/processed/objetivos_'+str(job)+'.pkl', 'rb') as f:
            objetivos.append(pickle.load(f))

        resultados[job] = [[] for i in range(len(resultadosX))]
        for i in range(len(resultados[job])):
            x = resultadosX[i]
            phi = np.zeros_like(x)
            phi[0] = x[0]
            for t in range(1, T):
                phi[t] = np.maximum(x[t]-x[t-1], 0)
            resultados[job][i] = list(resultadosX[i]) + list(phi)

    objetivos = np.array(objetivos)
    resultados = np.array(resultados)

    return A, b, c, resultados, objetivos

def get_coupling_constraints(X, instance, r=None):
    J = instance['jobs'][0]
    T = instance['tamanho'][0]
    uso_p = torch.Tensor(instance['uso_p']) # recurso utilizado por cada tarefa

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

    soc = torch.zeros_like(consumo)
    soc[:,0] = soc_inicial
    for t in range(1,T):
        soc[:,t] = soc[:,t-1] + (ef / q) * (i[:,t] / 60)

    g = torch.hstack((consumo - recurso_total, soc - 1, limite_inferior - soc))

    return g
