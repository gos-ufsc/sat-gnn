
jobs = 9


subs = 1 # número de máquinas/módulos
jobs = 9 # número de tarefas a serem escolonadas
recurso_p = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.]

tamanho = 97

priority = [5 2 1 4 1 3 1 1 1] # prioridade de cada tarefa
uso_p = [3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0] # recurso utilizado por cada tarefa
min_statup = [2 2 3 1 1 1 2 1 1] # tempo mínimo de vezes que uma tarefa pode iniciar
max_statup = [4 6 5 7 5 6 7 8 9] # tempo máximo de vezes que uma tarefa pode iniciar
min_cpu_time = [10 5 7 6 8 4 6 7 4] # tempo mínimo de unidades de tempo que uma tarefa pode consumir em sequência
max_cpu_time = [15 24 20 28 18 34 21 20 20] # tempo máximo de unidades de tempo que uma tarefa pode consumir em sequência
min_periodo_job = [30 28 24 32 24 28 24 24 24] # tempo mínimo que uma tarefa deve esperar para se repetir
max_periodo_job = [78 50 100 90 60 87 60 97 40] # tempo máximo que uma tarefa pode esperar para se repetir


win_min = [30 0 0 0 0 40 0 0 0]
win_max = [81 98 98 98 61 81 98 98 98]
