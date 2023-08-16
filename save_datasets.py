from pathlib import Path

from src.dataset import MultiTargetDataset, OptimalsDataset, OptimalsWithZetaDataset, VarOptimalityDataset


if __name__ == '__main__':
    # instances_fpaths = list(Path('data/raw/').glob('97_*.json'))

    # ds = MultiTargetDataset(instances_fpaths=instances_fpaths,
    #                         sols_dir='/home/bruno/sat-gnn/data/interim',
    #                         split='all')
    # ds.maybe_initialize()
    # ds.save_dataset('data/processed/multitarget_97_all.hdf5')

    # ds = VarOptimalityDataset(instances_fpaths=instances_fpaths,
    #                           sols_dir='/home/bruno/sat-gnn/data/interim',
    #                           split='all')
    # ds.maybe_initialize()
    # ds.save_dataset('data/processed/varoptimality_97_all.hdf5')

    # ds = OptimalsDataset(instances_fpaths=instances_fpaths,
    #                      sols_dir='/home/bruno/sat-gnn/data/interim',
    #                      split='all')
    # ds.maybe_initialize()
    # ds.save_dataset('data/processed/optimals_97_all.hdf5')

    # ds = OptimalsWithZetaDataset(instances_fpaths=instances_fpaths,
    #                              sols_dir='/home/bruno/sat-gnn/data/interim',
    #                              split='all')
    # ds.maybe_initialize()
    # ds.save_dataset('data/processed/optimalszeta_97_all.hdf5')

    ds = MultiTargetDataset(
        [fp for fp in Path('data/raw/').glob('125_*.json') if (int(fp.name.split('_')[1]) < 20)]
        + [fp for fp in Path('data/raw/').glob('97_*.json') if (int(fp.name.split('_')[1]) < 20)],
        split='all',
    )
    ds.maybe_initialize()
    ds.save_dataset('data/processed/multitarget_125_97_small_all_200.hdf5')

    # ds = MultiTargetDataset(
    #     [fp for fp in Path('data/raw/').glob('125_*.json') if int(fp.name.split('_')[1]) >= 20],
    #     split='all',
    # )
    # ds.maybe_initialize()
    # ds.save_dataset('data/processed/multitarget_125_large_all.hdf5')
