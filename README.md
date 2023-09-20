# GNNs for Nanosatellite Task Schedulling

[![DOI](http://img.shields.io/badge/cs.LG-arXiv%3A2303.13773-B31B1B.svg)](https://doi.org/10.48550/arXiv.2303.13773)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8356798.svg)](https://doi.org/10.5281/zenodo.8356798)

This repository implements the experiments described in the paper "Graph Neural Networks for the Offline Nanosatellite Task Scheduling Problem". All experiments were logged at [weights & biases](https://wandb.ai/brunompac/sat-gnn). Overall, the interesting bits of code can be found in `src/`, while `notebooks/` contain drafts, experiments and visualizations. The scripts in root are for generating the results we present in the paper.

Steps to reproduce results:

1. Get the data, either by
    - Generating new instances using `instance_generation.py` and then computing the solutions using `compute_solutions.py`; or
    - Downloading our [publicly available dataset](https://zenodo.org/record/8356798) (`instances/*.json` must go into `data/raw/`, while `solutions/*.npz` musto go into `data/interim`).
1. Maybe save the datasets into hdf5 files using `save_datasets.py`
    - This is useful if you don't want (or can't) keep the complete dataset in memory
1. Run the feasibility classification experiments using `feasibilty_classification_experiments.py`
1. Run the optimality prediction experiments
    1. First tune the models using `tune_multitarget.py` and `tune_optimals.py`
    2. Then train the best models found using `train_best_models.py`
1. Evaluate the SatGNN-based heuristics using `evaluate.py`
    - All of our evaluations are packed in `run_evaluation_test.sh` and `run_heuristic_experiments.sh`
1. Generate plots using `notebooks/6.0-bmp-paper.plots.ipynb`

Other useful scripts: `tmp_save_preds.py` saves the predictions for a fine-grained analysis; `remove_repeated_instances_and_rename.py` drops repeated instances when generating on multiple machines (probably outdated); `pred_feasibility` stores the resuts of the feasibility models; `bce_test_set.py` computes the test set loss without using the Trainer classes.
