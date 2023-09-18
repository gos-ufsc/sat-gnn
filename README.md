# GNNs for Nanosatellite Task Schedulling

See runs at https://wandb.ai/brunompac/sat-gnn.

Steps to reproduce results:

1. `instance_generation.py`
1. `compute_solutions.py`
    1. Maybe `save_datasets.py`
1. `feasibilty_classification_experiments.py`
1. `tune_multitarget.py` & `tune_optimals.py`
1. `train_best_models.py`
1. `evalute.py` with heuristic configurations of interest; or
    1. `run_evaluation_test.sh`
    1. `run_heuristic_experiments.sh`
1. Generate plots using `notebooks/6.0-bmp-paper.plots.ipynb`


Other useful scripts: `tmp_save_preds.py` saves the predictions for a fine-grained analysis; `remove_repeated_instances_and_rename.py` drops repeated instances when generating on multiple machines (probably outdated); `pred_feasibility` stores the resuts of the feasibility models; `bce_test_set.py` computes the test set loss without using the Trainer classes.
