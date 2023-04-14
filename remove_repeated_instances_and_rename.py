import json
from pathlib import Path
import numpy as np

def add_to_mbd(mbd, values):
    return mbd + '_'.join(str(v) for v in values)


if __name__ == '__main__':
    for T in [97, 120, 125, 154, 170, 194, 291]:
        for J in [9, 13, 18, 20, 22, 24]:
            fpaths = list(Path('/home/bruno/sat-gnn/data/raw').glob("%d_%d_*.json" % (T, J)))

            mbds = list()
            for fpath in fpaths:
                with open(fpath) as f:
                    instance = json.load(f)

                instance_mbd = "%d_%d_%f_%f" % (T, J, instance['power_use'][0], instance['power_resource'][0])

                instance_mbd = add_to_mbd(instance_mbd, instance['min_cpu_time'])
                instance_mbd = add_to_mbd(instance_mbd, instance['max_cpu_time'])
                instance_mbd = add_to_mbd(instance_mbd, instance['max_job_period'])
                instance_mbd = add_to_mbd(instance_mbd, instance['max_job_period'])
                instance_mbd = add_to_mbd(instance_mbd, instance['min_startup'])
                instance_mbd = add_to_mbd(instance_mbd, instance['max_startup'])
                instance_mbd = add_to_mbd(instance_mbd, instance['priority'])
                instance_mbd = add_to_mbd(instance_mbd, instance['win_min'])
                instance_mbd = add_to_mbd(instance_mbd, instance['win_max'])

                mbds.append(instance_mbd)

            fpaths = np.array(fpaths)

            uniques, inv, count = np.unique(mbds, return_inverse=True, return_counts=True)

            repeated = np.where(count > 1)[0]

            repeated_fpaths = list()
            for r in repeated:
                repeated_fpaths.append(fpaths[inv == r])

            # remove repeated
            for fps in repeated_fpaths:
                for fp in fps[1:]:  # keep first fp as the original
                    fp.unlink()

            unique_fpaths = list(Path('/home/bruno/sat-gnn/data/raw').glob("%d_%d_*.json" % (T, J)))
            unique_fpaths = np.array(unique_fpaths)

            # sort fpaths to avoid overwriting
            unique_fpaths_i = [int(fp.name[:-len('.json')].split('_')[-1]) for fp in unique_fpaths]
            unique_fpaths = unique_fpaths[np.argsort(unique_fpaths_i)]

            for i, fp in enumerate(unique_fpaths):
                fp.rename(fp.parent/("%d_%d_%d.json" % (T, J, i)))
