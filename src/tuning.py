from pathlib import Path
from typing import List

from optuna.trial import Trial
import torch

from src.net import SatGNN
from src.trainer import EarlyFixingTrainer


def get_objective(instances_fpaths: List[Path], opts, wandb_project=None,
                  wandb_group='EarlyFixingInstance-tuning2'):
    def objective(trial: Trial):
        egat_num_heads = None
        sage_aggregator_type = None
        sage_feat_drop = None

        # conv1 = trial.suggest_categorical('conv1', ['GraphConv', 'EGATConv', 'SAGEConv'])
        conv1 = trial.suggest_categorical('conv1', ['GraphConv', 'SAGEConv'])
        conv1_kwargs = dict()
        if conv1 == 'EGATConv':
            egat_num_heads = trial.suggest_int('EGAT_heads', 1, 5)
            conv1_kwargs['num_heads'] = egat_num_heads
        elif conv1 == 'SAGEConv':
            sage_aggregator_type = trial.suggest_categorical('SAGE_aggregator_type', ['pool', 'gcn', 'lstm'])
            sage_feat_drop = trial.suggest_float('SAGE_feat_drop', 0, 0.5)
            conv1_kwargs['aggregator_type'] = sage_aggregator_type
            conv1_kwargs['feat_drop'] = sage_feat_drop

        # conv2 = trial.suggest_categorical('conv2', [None, 'GraphConv', 'EGATConv', 'SAGEConv'])
        conv2 = trial.suggest_categorical('conv2', [None, 'GraphConv', 'SAGEConv'])
        conv2_kwargs = dict()
        if conv2 == 'EGATConv':
            if egat_num_heads is None:
                egat_num_heads = trial.suggest_int('EGAT_heads', 1, 5)
            conv2_kwargs['num_heads'] = egat_num_heads
        elif conv2 == 'SAGEConv':
            if sage_aggregator_type is None:
                sage_aggregator_type = trial.suggest_categorical('SAGE_aggregator_type', ['pool', 'gcn', 'lstm'])
                sage_feat_drop = trial.suggest_float('SAGE_feat_drop', 0, 0.5)
            conv2_kwargs['aggregator_type'] = sage_aggregator_type
            conv2_kwargs['feat_drop'] = sage_feat_drop

        # conv3 = trial.suggest_categorical('conv3', [None, 'GraphConv', 'EGATConv', 'SAGEConv'])
        conv3 = trial.suggest_categorical('conv3', [None, 'GraphConv', 'SAGEConv'])
        conv3_kwargs = dict()
        if conv3 == 'EGATConv':
            if egat_num_heads is None:
                egat_num_heads = trial.suggest_int('EGAT_heads', 1, 5)
            conv3_kwargs['num_heads'] = egat_num_heads
        elif conv3 == 'SAGEConv':
            if sage_aggregator_type is None:
                sage_aggregator_type = trial.suggest_categorical('SAGE_aggregator_type', ['pool', 'gcn', 'lstm'])
                sage_feat_drop = trial.suggest_float('SAGE_feat_drop', 0, 0.5)
            conv3_kwargs['aggregator_type'] = sage_aggregator_type
            conv3_kwargs['feat_drop'] = sage_feat_drop

        n_h_feats = trial.suggest_int('n_h_feats', 2, 20)
        single_conv_for_both_passes = trial.suggest_categorical(
            'single_conv_for_both_passes',
            [False, True]
        )
        n_passes = trial.suggest_int('n_passes', 1, 2)

        try:
            net = SatGNN(
                2,
                n_h_feats=n_h_feats,
                single_conv_for_both_passes=single_conv_for_both_passes,
                n_passes=n_passes,
                conv1=conv1, conv1_kwargs=conv1_kwargs,
                conv2=conv2, conv2_kwargs=conv2_kwargs,
                conv3=conv3, conv3_kwargs=conv3_kwargs,
                readout_op=None,
            )

            batch_power = trial.suggest_int('batch_power', 2, 7)
            samples_per_problem_power = trial.suggest_int('samples_per_problem_power', 6, 10)
            trainer = EarlyFixingTrainer(
                net,
                instances_fpaths=instances_fpaths,
                optimals=opts,
                samples_per_problem=2**samples_per_problem_power,
                batch_size=2**batch_power,
                epochs=20,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                timeout=2*60*60,
            )
            trainer.run()

            score = -max(trainer.val_scores[-5:])
        except torch.cuda.OutOfMemoryError:
            score = 0

        try:
            trial.set_user_attr('wandb_id', trainer._id)
        except:
            trial.set_user_attr('wandb_id', None)

        return score

    return objective
