import json
import os
import time

from multiprocessing import Process, Value, Array, Lock
from pathlib import Path
from time import sleep

import numpy as np
from gurobipy import GRB

from src.problem import get_model, random_instance


def save_instance(tjn, instance, target_dir):
    instance_fp = target_dir/("%d_%d_%d.json" % tjn)

    with open(instance_fp, 'w') as f:
        json.dump(instance, f)

def new_feasible_instance_or_none(t, j, timeout=10):
    instance = random_instance(t, j)

    model = get_model(instance, new_ineq=True, timeout=timeout)
    model.setObjective(1, GRB.MAXIMIZE)  # we just care about feasibility here
    model.update()
    model.optimize()

    if model.SolCount > 0:  # certainly feasible
        return instance
    else:
        return None

def instance_generator(shared, lock, target_dir, print_lock, name):
    # closest to a random seed we can get for a chid process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    while True:
        # get t and j
        with lock:
            i = shared['tj_i'].value

            # go to next instance size
            if shared['ns'][i] >= 100:
                i += 1
                if i >= len(shared['ts'][:]):
                    # all instances have already been processed
                    break
                else:
                    shared['tj_i'].value = i
                    with print_lock:
                        print(f"WORKER {name}: moving to {shared['ts'][i]}-{shared['js'][i]}")

            t = shared['ts'][i]
            j = shared['js'][i]

        # with print_lock:
        #     print('WORKER %d: new %d-%d' % (name, t, j))

        instance = new_feasible_instance_or_none(t, j)

        if instance is not None:
            with print_lock:
                print('WORKER %d: found feasible %d-%d' % (name, t, j))

            with lock:
                # get n
                ts = shared['ts'][:]
                js = shared['js'][:]
                tjs = list(zip(ts, js))
                n_i = tjs.index((t,j))
                n = shared['ns'][n_i]

                with print_lock:
                    print('WORKER %d: saving %d-%d-%d' % (name, t, j, n))

                # save instance
                save_instance((t,j,n), instance, target_dir)

                # increase n
                shared['ns'][n_i] += 1
        else:
            with print_lock:
                print('WORKER %d: infeasible %d-%d' % (name, t, j))

# q=5, soc_inicial=0.7, limite_inferior=0.0,
#                     bat_usage=5, ef=0.9, k=1, beta=10, v_bat=3.6, 

if __name__ == '__main__':
    n_processes = 12
    target_dir = Path('/home/bruno/sat-gnn/data/raw/')

    # get number of instances by size
    new_instances = dict()
    tjs = list()
    ns = list()
    for tj in [(t, j) for t in [97, 120, 125, 154, 170, 194, 291]
                      for j in [9, 18, 20, 22, 24]]:
        n = len(list(target_dir.glob("%d_%d_*.json" % tj)))

        if n < 100:
            tjs.append(tj)
            ns.append(n)
            new_instances[tj] = n

    shared = {
        'ts': Array('i', [t for t, _ in tjs]),
        'js': Array('i', [j for _, j in tjs]),
        'tj_i': Value('i', 0),
        'ns': Array('i', ns),
    }
    lock = Lock()
    print_lock = Lock()

    ps = [Process(target=instance_generator, args=(shared, lock, target_dir, print_lock, name))
          for name in range(n_processes)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
