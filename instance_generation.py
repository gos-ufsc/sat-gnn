import json
import os
import time
import logging
import logging.handlers

from multiprocessing import Process, Value, Array, Lock, Queue, current_process
from pathlib import Path
from time import sleep

import numpy as np
from gurobipy import GRB

from src.problem import Instance


def listener_configurer(fpath):
    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler(fpath, 'a', 3_000_000, 10)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)

def listener_process(queue, log_fpath):
    listener_configurer(log_fpath)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)

def save_instance(tjn, instance: Instance, target_dir):
    instance_fp = target_dir/("%d_%d_%d.json" % tjn)
    instance.to_json(instance_fp)

def new_feasible_instance_or_none(t, j, timeout=60):
    instance = Instance.random(t, j)

    model = instance.to_gurobipy(new_inequalities=True, timeout=timeout)
    model.setObjective(1, GRB.MAXIMIZE)  # we just care about feasibility here
    model.update()
    model.optimize()

    if model.SolCount > 0:  # certainly feasible
        return instance
    else:
        return None

def instance_generator(shared, lock, target_dir, log_queue, N=100):
    # closest to a random seed we can get for a chid process
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    worker_configurer(log_queue)
    name = current_process().name
    logger = logging.getLogger('GEN')
    logger.debug('Worker %s started' % name)

    while True:
        # get t and j
        with lock:
            i = shared['tj_i'].value

            # go to next instance size
            if shared['ns'][i] >= N:
                i += 1
                if i >= len(shared['ts'][:]):
                    # all instances have already been processed
                    break
                else:
                    shared['tj_i'].value = i
                    logger.debug(f"moving to {shared['ts'][i]}-{shared['js'][i]}")

            t = shared['ts'][i]
            j = shared['js'][i]

        # with print_lock:
        #     print('WORKER %d: new %d-%d' % (name, t, j))

        instance = new_feasible_instance_or_none(t, j)

        if instance is not None:
            logger.debug('found feasible %d-%d' % (t, j))

            with lock:
                # get n
                ts = shared['ts'][:]
                js = shared['js'][:]
                tjs = list(zip(ts, js))
                n_i = tjs.index((t,j))
                n = shared['ns'][n_i]

                logger.debug('saving %d-%d-%d' % (t, j, n))

                # save instance
                save_instance((t,j,n), instance, target_dir)

                # increase n
                shared['ns'][n_i] += 1
        else:
            logger.debug('infeasible %d-%d' % (t, j))


if __name__ == '__main__':
    n_processes = 12
    target_dir = Path('/home/bruno/sat-gnn/data/raw/new')

    N = 200

    # get number of instances by size
    new_instances = dict()
    tjs = list()
    ns = list()
    for tj in [(t, j) for t in [125,97]
                      for j in [9,13,18]]:
        n = len(list(target_dir.glob("%d_%d_*.json" % tj)))

        if n < N:
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

    # logging setup
    log_queue = Queue(-1)
    log_fpath = target_dir/'instance_generation.log'
    listener = Process(target=listener_process, args=(log_queue, log_fpath))

    ps = [Process(target=instance_generator, args=(shared, lock, target_dir,
                                                   log_queue, N))
          for _ in range(n_processes)]

    listener.start()
    for p in ps:
        p.start()

    for p in ps:
        p.join()
