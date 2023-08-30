from pathlib import Path

from src.dataset import MultiTargetDataset, OptimalsDataset


if __name__ == '__main__':
    instances_fpaths = [fp for fp in Path('data/raw/').glob('125_*.json')
                        if (int(fp.name.split('_')[1]) < 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) < 200)]

   #  ds = MultiTargetDataset(instances_fpaths=instances_fpaths,
   #                          sols_dir='/home/bruno/sat-gnn/data/interim')
   #  ds.maybe_initialize()
   #  ds.save_dataset('data/processed/multitarget_125_train.hdf5')

    ds = OptimalsDataset(instances_fpaths=instances_fpaths,
                         sols_dir='/home/bruno/sat-gnn/data/interim')
    ds.maybe_initialize()
    ds.save_dataset('data/processed/optimals_125_train.hdf5')

    instances_fpaths = [fp for fp in Path('data/raw/').glob('125_*.json')
                        if (int(fp.name.split('_')[1]) >= 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) < 20)]

   #  ds = MultiTargetDataset(instances_fpaths=instances_fpaths,
   #                          sols_dir='/home/bruno/sat-gnn/data/interim')
   #  ds.maybe_initialize()
   #  ds.save_dataset('data/processed/multitarget_125_val.hdf5')

    ds = OptimalsDataset(instances_fpaths=instances_fpaths,
                         sols_dir='/home/bruno/sat-gnn/data/interim')
    ds.maybe_initialize()
    ds.save_dataset('data/processed/optimals_125_val.hdf5')
