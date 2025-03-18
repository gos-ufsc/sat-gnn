import logging
import random
from abc import ABC
from contextlib import nullcontext
from pathlib import Path
from time import time

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.data import DGLDataset
from dgl.data.utils import Subset
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from src.net import SatGNN
from src.problem import ModelWithPrimalDualIntegral
from src.utils import timeit


class Trainer(ABC):
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, training_dataset: Dataset,
                 validation_dataset: Dataset = None,
                 test_dataset: Dataset = None, get_best_model=False, epochs=5,
                 lr= 0.1, batch_size=2**4, optimizer: str = 'Adam',
                 optimizer_params=dict(), loss_func: str = 'MSELoss',
                 loss_func_params=dict(), lr_scheduler: str = None,
                 lr_scheduler_params=dict(), mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        self._is_initalized = False

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.loss_func_params = loss_func_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self._dtype = next(self.net.parameters()).dtype

        self.mixed_precision = mixed_precision

        if logger is None:
            logging.basicConfig(level=logging.INFO, datefmt='%y/%m/%d--%H:%M:%S')
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.checkpoint_every = checkpoint_every

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self._log_to_wandb = False if wandb_project is None else True
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

        self.max_loss = max_loss
        self.get_best_model = get_best_model

        self.l.info('Preparing data')
        self.train_data, self.val_data, self.test_data = self.prepare_data(
            training_dataset,
            validation_dataset,
            test_dataset
        )

    def _load_optim(self, state_dict=None):
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        if state_dict is not None:
            self._optim.load_state_dict(state_dict)
    
    def _load_lr_scheduler(self, state_dict=None):
        Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
        self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        if state_dict is not None:
            self._scheduler.load_state_dict(state_dict)

    def _load_loss_func(self):
        LossClass = eval(f"nn.{self.loss_func}")
        self._loss_func = LossClass(**self.loss_func_params)

    @classmethod
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project=None,
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and creates the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            batch_size=wandb.config['batch_size'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        self._load_optim(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            self._load_lr_scheduler(state_dict=checkpoint['scheduler_state_dict'])

        self._load_loss_func(self)

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.train_data, self.val_data = self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        self._load_optim(state_dict=None)

        if self.lr_scheduler is not None:
            self._load_lr_scheduler(state_dict=None)

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "mixed_precision": self.mixed_precision,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            })

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._load_loss_func()

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self._is_initalized = True

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            job_type='train',
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net, log=None)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    def prepare_data(self, training_dataset, validation_dataset=None,
                     test_dataset=None):
        """Instantiate data loaders.
        """
        train_loader = DataLoader(training_dataset, self.batch_size, shuffle=True)

        if validation_dataset is not None:
            val_loader = DataLoader(validation_dataset, self.batch_size, shuffle=False)
        else:
            val_loader = None

        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)
        else:
            test_loader = None

        return train_loader, val_loader, test_loader

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix+k] = v
        
        return data_to_log

    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        data_to_log = {
            "train_loss": train_losses['all'],
            "train_time": train_time,
        }

        self._add_data_to_log(train_losses, 'train_loss_', data_to_log)
        self._add_data_to_log(train_times, 'train_time_', data_to_log)

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        if self.val_data is not None:
            # validation
            val_time, (val_losses, val_times) = timeit(self.validation_pass)()

            self.l.info(f"Validation pass took {val_time:.3f} seconds")
            self.l.info(f"Validation loss = {val_losses['all']}")

            data_to_log["val_loss"] = val_losses['all']
            data_to_log["val_time"] =val_time

            self._add_data_to_log(val_losses, 'val_loss_', data_to_log)
            self._add_data_to_log(val_times, 'val_time_', data_to_log)

            val_score = val_losses['all']  # defines best model

            if val_score < self.best_val:
                self.best_val = val_score

                if self.get_best_model:
                    if self._log_to_wandb:
                        self.l.info(f"Saving best model")
                        self.save_model(name='model_best')

                    if self.test_data is not None:
                        # test
                        test_losses, test_times = self.test_pass()

                        self.l.info(f"Test loss = {test_losses['all']}")

                        data_to_log["test_loss"] = test_losses['all']

                        self._add_data_to_log(test_losses, 'test_loss_', data_to_log)
                        self._add_data_to_log(test_times, 'test_time_', data_to_log)

        return data_to_log

    def run(self) -> nn.Module:
        if not self._is_initalized:
            self.setup_training()

        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            if self.early_stopping(data_to_log):
                break

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

        # evaluating final model
        if self.test_data is not None:
            # test
            test_losses, _ = self.test_pass()

            self.l.info(f"Final model's test loss = {test_losses['all']}")

            if self._log_to_wandb:
                # wandb.log(
                #     {'last_test_'+k: v for k, v in test_losses.items() if k != 'all'},
                #     step=self._e, commit=True,
                # )
                for k, v in test_losses.items():
                    if k.startswith('table_') or k.startswith('plot_'):
                        wandb.log({
                            'last_test_'+k: v
                        }, step=self._e)
                    elif k != 'all':
                        wandb.run.summary['last_test_'+k] = v

        if self._log_to_wandb:
            wandb.run.summary['best_val'] = self.best_val

            wandb.finish()

        self.l.info('Training finished!')

        return self.net
    
    def early_stopping(self, data_to_log):
        """Return True if the early-stopping criteria is met.
        """
        try:
            return data_to_log['val_loss'] > self.max_loss
        except:
            return False

    def optim_step(self, loss):
        if self.mixed_precision:
            backward_time, _  = timeit(self._scaler.scale(loss).backward)()
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            backward_time, _  = timeit(loss.backward)()
            self._optim.step()

        self._optim.zero_grad()
        return backward_time
    
    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat.view_as(y), y.to(y_hat))

        metrics = None
        if validation:
            # here you can compute performance metrics (populate `metrics`)
            pass

        return loss_time, loss, metrics
    
    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        return losses

    def data_pass(self, data, train=False):
        loss = 0
        size = 0
        metrics = list()

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(train):
            for X, y in data:
                X = X.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    batch_forward_time, y_hat = timeit(self.net)(X)
                    forward_time += batch_forward_time

                    batch_loss_time, batch_loss, batch_metrics = self.get_loss_and_metrics(y_hat, y, validation=not train)
                loss_time += batch_loss_time

                metrics.append(batch_metrics)

                if train:
                    batch_backward_time = self.optim_step(batch_loss)
                    backward_time += batch_backward_time

                loss += batch_loss.item() * len(y)
                size += len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        if all([b_m is None for b_m in metrics]):
            metrics = None

        losses = self.aggregate_loss_and_metrics(loss, size, metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    def train_pass(self):
        self.net.train()
        return self.data_pass(self.train_data, train=True)

    def validation_pass(self):
        self.net.eval()
        return self.data_pass(self.val_data, train=False)

    def test_pass(self):
        self.net.eval()
        return self.data_pass(self.test_data, train=False)

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath

class GraphTrainer(Trainer):
    def __init__(self, net: SatGNN, training_dataset: DGLDataset,
                 validation_dataset: DGLDataset = None,
                 test_dataset: DGLDataset = None, get_best_model=False,
                 epochs=5, lr=1e-3, batch_size=2 ** 4,
                 optimizer: str = 'Adam', optimizer_params=dict(),
                 loss_func: str = 'BCEWithLogitsLoss', loss_func_params=dict(),
                 lr_scheduler: str = None, lr_scheduler_params=dict(),
                 mixed_precision=True, device=None, wandb_project=None,
                 wandb_group=None, logger=None, checkpoint_every=50,
                 random_seed=42, max_loss=None) -> None:
        super().__init__(net, training_dataset, validation_dataset,
                         test_dataset, get_best_model, epochs, lr, batch_size, optimizer,
                         optimizer_params, loss_func, loss_func_params,
                         lr_scheduler, lr_scheduler_params, mixed_precision,
                         device, wandb_project, wandb_group, logger,
                         checkpoint_every, random_seed, max_loss)


        self._add_to_wandb_config({
            "n_passes": self.net.n_passes,
            "n_h_feats": self.net.n_h_feats,
            "n_output_layers": self.net.n_output_layers,
            "single_conv": self.net.single_conv_for_both_passes,
            "n_convs": len(self.net.convs),
            "train_dataset": training_dataset.name,
            "val_dataset": validation_dataset.name if validation_dataset is not None else None,
            "test_dataset": test_dataset.name if test_dataset is not None else None,
        })

    def prepare_data(self, training_dataset, validation_dataset=None,
                     test_dataset=None):
        """Instantiate data loaders.
        """
        train_loader = GraphDataLoader(training_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        if validation_dataset is not None:
            val_loader = GraphDataLoader(validation_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False)
        else:
            val_loader = None

        if test_dataset is not None:
            # does not use a dataloader for early-fixing evaluation
            test_loader = test_dataset
        else:
            test_loader = None

        return train_loader, val_loader, test_loader

    def data_pass(self, data, train=False):
        loss = 0
        size = 0
        metrics = list()

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train(train)
        with torch.set_grad_enabled(train):
            for g in data:
                g = g.to(self.device)

                with self.autocast_if_mp():
                    batch_forward_time, output = timeit(self.net)(g)
                    forward_time += batch_forward_time

                    batch_loss_time, batch_loss, batch_metrics = self.get_loss_and_metrics(output, g, validation=not train)
                loss_time += batch_loss_time

                metrics.append(batch_metrics)

                if train:
                    batch_backward_time = self.optim_step(batch_loss)
                    backward_time += batch_backward_time

                loss += batch_loss.item() * g.batch_size
                size += g.batch_size

            if self.lr_scheduler is not None:
                self._scheduler.step()

        if all([b_m is None for b_m in metrics]):
            metrics = None

        losses = self.aggregate_loss_and_metrics(loss, size, metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times

    @staticmethod
    def evaluate_early_fixing(model, fixed_vars: dict = None, timeout=10,
                              hide_output=True):
        model_ = ModelWithPrimalDualIntegral(sourceModel=model)
        model_.setParam('limits/time', timeout)
        model_.hideOutput(hide_output)

        if fixed_vars is not None:
            for var in model_.getVars():
                try:
                    fixed_var_X = fixed_vars[var.name]
                    model_.fixVar(var, fixed_var_X)
                except KeyError:
                    pass

        model_.optimize()

        if model_.getStatus().lower() not in ['optimal', 'timelimit']:
            infeasible = True
            runtime = model_.getSolvingTime()
            objective = 0
            gap = -1
            primal_dual_integral = -1
        else:
            infeasible = False
            runtime = model_.getSolvingTime()
            try:
                objective = model_.getObjVal()
                gap = model_.getGap()
                primal_dual_integral = model_.get_primal_dual_integral()
            except:
                # in case the problem is not infeasible but not solution was
                # found during the time limit
                objective = 0
                gap = -1
                primal_dual_integral = -1

        return infeasible, runtime, objective, gap, primal_dual_integral

class FeasibilityClassificationTrainer(GraphTrainer):
    def data_pass(self, data, train=False):
        loss = 0
        size = 0
        metrics = list()

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train(train)
        with torch.set_grad_enabled(train):
            for g, y in tqdm(data):
                g = g.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    batch_forward_time, output = timeit(self.net)(g)
                    forward_time += batch_forward_time

                    batch_loss_time, batch_loss, batch_metrics = self.get_loss_and_metrics(output, y, validation=not train)
                loss_time += batch_loss_time

                metrics.append(batch_metrics)

                if train:
                    batch_backward_time = self.optim_step(batch_loss)
                    backward_time += batch_backward_time

                loss += batch_loss.item() * g.batch_size
                size += g.batch_size

            if self.lr_scheduler is not None:
                self._scheduler.step()

        if all([b_m is None for b_m in metrics]):
            metrics = None

        losses = self.aggregate_loss_and_metrics(loss, size, metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times
    
    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat.view_as(y), y.to(y_hat))

        metrics = None
        if validation:
            pred = torch.sigmoid(y_hat.view_as(y)) > 0.5
            pred = pred.to(y)
            hit = pred == y

            metrics = (hit,)

        return loss_time, loss, metrics

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        if metrics is not None:
            hits = [m[0] for m in metrics]
            hits = torch.hstack(hits)

            losses['accuracy'] = hits.sum().item() / size

        return losses

class InstanceFeasibilityClassificationTrainer(FeasibilityClassificationTrainer):
    def __init__(self, net: SatGNN, dataset: DGLDataset,
                 test_dataset: DGLDataset = None, get_best_model=False,
                 epochs=5, lr=0.001, batch_size=2 ** 7,
                 optimizer: str = 'Adam', optimizer_params=dict(),
                 loss_func: str = 'BCEWithLogitsLoss', loss_func_params=dict(),
                 lr_scheduler: str = None, lr_scheduler_params=dict(),
                 mixed_precision=True, device=None, wandb_project=None,
                 wandb_group=None, logger=None, checkpoint_every=50,
                 random_seed=42, max_loss=None, min_train_loss=1e-2) -> None:
        super().__init__(net, dataset, None, test_dataset, get_best_model,
                         epochs, lr, batch_size, optimizer,
                         optimizer_params, loss_func, loss_func_params,
                         lr_scheduler, lr_scheduler_params, mixed_precision,
                         device, wandb_project, wandb_group, logger,
                         checkpoint_every, random_seed, max_loss)
        self.min_train_loss = min_train_loss

    def prepare_data(self, dataset, validation_dataset=None, test_dataset=None):
        n = len(dataset)
        all_idx = np.arange(n)

        training_idx = np.random.choice(all_idx, int(0.8*n), replace=False)
        training_dataset = Subset(dataset, training_idx)

        validation_idx = [i for i in all_idx if i not in training_idx]
        validation_dataset = Subset(dataset, validation_idx)

        return super().prepare_data(training_dataset, validation_dataset,
                                    test_dataset)

    def early_stopping(self, data_to_log):
        return data_to_log['train_loss'] < self.min_train_loss

class MultiTargetTrainer(GraphTrainer):
    def __init__(self, net: SatGNN, training_dataset: DGLDataset,
                 validation_dataset: DGLDataset = None,
                 test_dataset: DGLDataset = None, get_best_model=True,
                 epochs=5, lr=0.001,
                 optimizer: str = 'Adam', optimizer_params=dict(),
                 loss_func: str = 'BCEWithLogitsLoss',
                 loss_func_params={'reduction': 'none'},
                 lr_scheduler: str = None, lr_scheduler_params=dict(),
                 mixed_precision=True, device=None, wandb_project=None,
                 wandb_group=None, logger=None, checkpoint_every=50,
                 random_seed=42, max_loss=None) -> None:
        batch_size = 1
        super().__init__(net, training_dataset, validation_dataset,
                         test_dataset, get_best_model, epochs,
                         lr, batch_size, optimizer, optimizer_params,
                         loss_func, loss_func_params, lr_scheduler,
                         lr_scheduler_params, mixed_precision, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

    def get_loss_and_metrics(self, y_hat, g, validation=False):
        y = g.ndata['y']['var'].to(y_hat)
        w = g.ndata['w']['var'].to(y_hat)

        #compute weight
        weight = torch.softmax(w / w.max(-1)[0].unsqueeze(-1), -1)

        loss_time, loss =  timeit(self._loss_func)(
            y_hat.repeat(*(np.array(y.shape) // np.array(y_hat.shape))),
            y.to(y_hat),
        )
        loss = (weight * loss).mean(0).sum()

        metrics = None
        if validation:
            y_pred_ = (torch.sigmoid(y_hat) > 0.5).squeeze(0).cpu().numpy().astype(int)
            y_ = (y.cpu().numpy()[:,:len(y_pred_)] > 0.5).astype(int)
            hit = y_pred_.tolist() in y_.tolist()
            if hit:
                hit_i = y_.tolist().index(y_pred_.tolist())
                gap = w.max() - w[hit_i]
            else:
                gap = -1
            # here you can compute validation-specific metrics
            metrics = (hit, gap)

        return loss_time, loss, metrics

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            hits = [m[0] for m in metrics]
            gaps = [m[1] for m in metrics]

            acc = sum(hits) / size
            losses['accuracy'] = acc
            try:
                losses['mean_gap'] = sum(g for g in gaps if g != -1) / sum(1 for g in gaps if g != -1)
            except ZeroDivisionError:
                losses['mean_gap'] = torch.inf

        return losses

class OptimalsTrainer(GraphTrainer):
    def __init__(self, net: SatGNN, training_dataset: DGLDataset,
                 validation_dataset: DGLDataset = None,
                 test_dataset: DGLDataset = None, get_best_model=False,
                 epochs=5, lr=0.01, batch_size=2**2, optimizer: str = 'Adam',
                 optimizer_params=dict(), loss_func: str = 'BCEWithLogitsLoss',
                 loss_func_params=dict(), lr_scheduler: str = None,
                 lr_scheduler_params=dict(), mixed_precision=True, device=None,
                 wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        super().__init__(net, training_dataset, validation_dataset,
                         test_dataset, get_best_model, epochs,
                         lr, batch_size, optimizer, optimizer_params,
                         loss_func, loss_func_params, lr_scheduler,
                         lr_scheduler_params, mixed_precision, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed, max_loss)

    # def prepare_data(self, dataset):
    def prepare_data(self, training_dataset, validation_dataset=None,
                     test_dataset=None):
        train_loader, val_loader, test_loader = super().prepare_data(
            training_dataset,
            validation_dataset,
            test_dataset
        )

        # TODO: move this out of the trainer?
        # pre-training
        self.net.eval()
        self.net.pretrain = True
        with torch.no_grad():
            for g in train_loader:
                # run through all instances to update internal values
                _ = self.net(g.to(self.device))
        self.net.pretrain = False

        return train_loader, val_loader, test_loader

    def get_loss_and_metrics(self, y_hat, g, validation=False):
        y = g.ndata['y']['var']
        loss_time, loss =  timeit(self._loss_func)(y_hat.view_as(y), y)

        metrics = None
        if validation:
            y_preds = [g_.nodes['var'].data['logit'] for g_ in dgl.unbatch(g)]
            y_preds = [(torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(int) for y_pred in y_preds]

            ys = [g_.nodes['var'].data['y'] for g_ in dgl.unbatch(g)]
            ys = [(y.cpu().numpy() > 0.5).astype(int) for y in ys]

            accs = [(y_pred.reshape(y.shape) == y).sum() / y.shape[-1] for y, y_pred in zip(ys, y_preds)]

            metrics = np.array(accs)

        return loss_time, loss, metrics

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            accs = np.hstack(metrics)

            losses['mean_acc'] = np.mean(accs)
            losses['acc'] = accs

        return losses
