import logging
import random
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from time import time
from typing import List

import dgl
import gurobipy
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.sampler import SubsetRandomSampler

from src.problem import get_soc, get_model, load_data, load_instance
from src.dataset import InstanceEarlyFixingDataset, OnlyXInstanceEarlyFixingDataset, ResourceDataset, SatsDataset, VarClassDataset
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
    def __init__(self, net: nn.Module, epochs=5, lr= 0.1,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=True,
                 device=None, wandb_project=None, wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42,
                 max_loss=None, timeout=np.inf) -> None:
        self._is_initalized = False

        self.timeout = timeout

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        self._dtype = next(self.net.parameters()).dtype

        self.mixed_precision = mixed_precision

        if logger is None:
            logging.basicConfig(level=logging.INFO)
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

        self._val_score_label = 'all'

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
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        self._load_optim()

        if self._log_to_wandb:
            self._add_to_wandb_config({
                "learning_rate": self.lr,
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

        self._loss_func = eval(f"nn.{self.loss_func}()")

        if self.mixed_precision:
            self._scaler = GradScaler()
            self.autocast_if_mp = autocast
        else:
            self.autocast_if_mp = nullcontext

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def _load_optim(self):
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

    def _add_to_wandb_config(self, d: dict):
        if not hasattr(self, '_wandb_config'):
            self._wandb_config = dict()

        for k, v in d.items():
            self._wandb_config[k] = v

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    @abstractmethod
    def prepare_data(self):
        """Must populate `self.data` and `self.val_data`.
        """
        # TODO: maybe I should refactor this so that the Dataset is an input to
        # the Trainer?

    @staticmethod
    def _add_data_to_log(data: dict, prefix: str, data_to_log=dict()):
        for k, v in data.items():
            if k != 'all':
                data_to_log[prefix+k] = v
        
        return data_to_log

    def _run_epoch(self):
        # train
        train_time, (train_losses, train_times) = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_losses['all']}")

        # validation
        val_time, (val_losses, val_times) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_losses['all']}")

        data_to_log = {
            "train_loss": train_losses['all'],
            "val_loss": val_losses['all'],
            "train_time": train_time,
            "val_time": val_time,
        }
        self._add_data_to_log(train_losses, 'train_loss_', data_to_log)
        self._add_data_to_log(val_losses, 'val_loss_', data_to_log)
        self._add_data_to_log(train_times, 'train_time_', data_to_log)
        self._add_data_to_log(val_times, 'val_time_', data_to_log)

        val_score = val_losses[self._val_score_label]  # defines best model

        return data_to_log, val_score

    def run(self) -> nn.Module:
        if not self._is_initalized:
            self.setup_training()

        self.val_scores = list()

        start = time()
        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log, val_score = self._run_epoch()

            self.val_scores.append(val_score)

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            if val_score < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = val_score

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            if self.max_loss is not None:
                if val_score > self.max_loss:
                    break

            if time() - start > self.timeout:
                break

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

        return self.net

    def train_pass(self):
        train_loss = 0
        train_size = 0

        forward_time = 0
        loss_time = 0
        backward_time = 0

        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in self.data:
                X = X.to(self.device)
                y = y.to(self.device)

                self._optim.zero_grad()

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss = self.get_loss_and_metrics(y_hat, y)
                    loss_time += loss_time_

                if self.mixed_precision:
                    backward_time_, _  = timeit(self._scaler.scale(loss).backward)()
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    backward_time_, _  = timeit(loss.backward)()
                    self._optim.step()
                backward_time += backward_time_

                train_loss += loss.item() * len(y)
                train_size += len(y)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        losses = self.aggregate_loss_and_metrics(train_loss, train_size)
        times = {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
        }

        return losses, times
    
    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat, y)

        if validation:
            # here you can compute performance metrics
            return loss_time, loss, None
        else:
            return loss_time, loss
    
    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        return losses

    def validation_pass(self):
        val_loss = 0
        val_size = 0
        val_metrics = list()

        forward_time = 0
        loss_time = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self.val_data:
                X = X.to(self.device)
                y = y.to(self.device)

                with self.autocast_if_mp():
                    forward_time_, y_hat = timeit(self.net)(X)
                    forward_time += forward_time_

                    loss_time_, loss, metrics = self.get_loss_and_metrics(y_hat, y, validation=True)
                    loss_time += loss_time_

                    val_metrics.append(metrics)

                val_loss += loss.item() * len(y)  # scales to data size
                val_size += len(y)

        losses = self.aggregate_loss_and_metrics(val_loss, val_size, val_metrics)
        times = {
            'forward': forward_time,
            'loss': loss_time,
        }

        return losses, times

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

class JobFeasibilityTrainer(Trainer):
    def __init__(self, net: nn.Module, instance_fpath="data/raw/97_9.jl",
                 epochs=5, lr=0.001, batch_size: int = 2**4,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 loss_func: str = 'BCEWithLogitsLoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=False,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        super().__init__(net, epochs, lr, optimizer, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params,
                         mixed_precision, device, wandb_project, wandb_group,
                         logger, checkpoint_every, random_seed, max_loss)

        self.instance_fpath = Path(instance_fpath)
        self.batch_size = batch_size

        self._add_to_wandb_config({
            "instance": self.instance_fpath.name,
            "batch_size": self.batch_size,
        })

    def prepare_data(self):
        data = SatsDataset(self.instance_fpath)

        n_train = 8000  # leave last job for testing

        train_sampler = SubsetRandomSampler(torch.arange(n_train))
        test_sampler = SubsetRandomSampler(torch.arange(n_train, len(data)))

        self.data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )
        self.val_data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=test_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )

    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat, y.unsqueeze(-1))

        if validation:
            y_pred = (torch.sigmoid(y_hat) > 0.5).squeeze(1).cpu().numpy().astype(int)
            hits = sum(y_pred == y.cpu().numpy())
            # here you can compute validation-specific metrics
            return loss_time, loss, hits
        else:
            return loss_time, loss

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            losses['accuracy'] = sum(metrics) / size

        return losses

class EarlyFixingTrainer(Trainer):
    def __init__(self, net: nn.Module, instances_fpaths: List[Path],
                 optimals: Path, epochs=5, lr=0.001, batch_size: int = 2 ** 4,
                 samples_per_problem: int = 1000, optimizer: str = 'Adam',
                 optimizer_params: dict = None, n_instances_for_test=2,
                 loss_func: str = 'BCEWithLogitsLoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=False,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None, timeout=np.inf) -> None:
        super().__init__(
            net, epochs, lr, optimizer, optimizer_params, loss_func,
            lr_scheduler, lr_scheduler_params, mixed_precision, device,
            wandb_project, wandb_group, logger, checkpoint_every, random_seed,
            max_loss, timeout,
        )

        assert len(optimals) >= len(instances_fpaths)

        self.instances_fpaths = [Path(i) for i in instances_fpaths]
        self.batch_size = batch_size
        self.samples_per_problem = samples_per_problem
        self.optimals = optimals
        self.n_instances_for_test = int(n_instances_for_test)

        self._add_to_wandb_config({
            "instances": [i.name for i in instances_fpaths],
            "batch_size": self.batch_size,
            "samples_per_problem": self.samples_per_problem,
            "n_instances_for_test": self.n_instances_for_test,
            "n_passes": self.net.n_passes,
            "n_h_feats": self.net.n_h_feats,
            "single_conv": self.net.single_conv_for_both_passes,
            "n_convs": len(self.net.convs),
        })

        self._val_score_label = 'accuracy'

    def prepare_data(self):
        data = InstanceEarlyFixingDataset(
            [load_instance(i) for i in self.instances_fpaths],
            [self.optimals[i.name]['sol'] for i in self.instances_fpaths],
            samples_per_problem=self.samples_per_problem,
        )

        # leave last job for testing
        train_sampler = SubsetRandomSampler(torch.arange(
            self.samples_per_problem * (len(self.instances_fpaths)
                                        - self.n_instances_for_test)
        ))
        test_sampler = SubsetRandomSampler(torch.arange(
            self.samples_per_problem * (len(self.instances_fpaths)
                                        - self.n_instances_for_test),
            self.samples_per_problem * len(self.instances_fpaths)
        ))

        self.data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=False,
        )
        self.val_data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=test_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )

    def get_loss_and_metrics(self, y_hat, y, validation=False):
        loss_time, loss =  timeit(self._loss_func)(y_hat, y.float())

        if validation:
            y_pred = (torch.sigmoid(y_hat) > 0.5).squeeze(1).cpu().numpy().astype(int)
            hits = sum(y_pred == y.cpu().numpy())
            # here you can compute validation-specific metrics
            return loss_time, loss, hits
        else:
            return loss_time, loss

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
        }

        if metrics is not None:
            acc = sum(metrics) / size
            losses['accuracy'] = acc.mean()
            losses['accuracy_per_dimension'] = acc

        return losses

class OnlyXEarlyFixingInstanceTrainer(EarlyFixingTrainer):
    def prepare_data(self):
        data = OnlyXInstanceEarlyFixingDataset(
            [load_instance(i) for i in self.instances_fpaths],
            [self.optimals[i.name]['sol'].reshape((9,97*2))[:,:97].flatten() for i in self.instances_fpaths],
            samples_per_problem=self.samples_per_problem,
        )

        # leave last job for testing
        train_sampler = SubsetRandomSampler(torch.arange(
            self.samples_per_problem * (len(self.instances_fpaths)
                                        - self.n_instances_for_test)
        ))
        test_sampler = SubsetRandomSampler(torch.arange(
            self.samples_per_problem * (len(self.instances_fpaths)
                                        - self.n_instances_for_test),
            self.samples_per_problem * len(self.instances_fpaths)
        ))

        self.data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=False,
        )
        self.val_data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=test_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )

class VariableResourceTrainer(EarlyFixingTrainer):
    def __init__(self, net: nn.Module, instance_fpath="data/raw/97_9.jl",
                 epochs=5, lr=0.001, batch_size: int = 2 ** 4, mu0 = 1, lambda0=0,
                 samples_per_problem: int = 1000, optimizer: str = 'Adam',
                 optimizer_params: dict = None, lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, mixed_precision=False,
                 device=None, wandb_project=None, wandb_group=None, logger=None,
                 checkpoint_every=50, random_seed=42, max_loss=None) -> None:
        super().__init__(net, instance_fpath, epochs, lr, batch_size,
                         samples_per_problem, optimizer, optimizer_params,
                         'MSELoss', lr_scheduler, lr_scheduler_params,
                         mixed_precision, device, wandb_project, wandb_group,
                         logger, checkpoint_every, random_seed, max_loss)

        self.mu0 = mu0
        self.lambda0 = lambda0

    def prepare_data(self):
        # define problem
        self.instance = load_instance(self.instance_fpath)

        data = ResourceDataset(self.instance, n_samples=self.samples_per_problem)

        # AugLag parameters
        self.lambdak = dict()
        self.muk = self.mu0
        self.best_cons_for_update = torch.inf

        self.model_data = dict()
        for model, r in zip(data.models, data.rs):
            r_i = tuple(r.tolist())

            A = model.getA().toarray()
            b = np.array(model.getAttr('rhs'))
            c = np.array(model.getAttr('obj'))

            constraints_sense = np.array([ci.sense for ci in model.getConstrs()])

            A[constraints_sense == '<'] *= -1
            b[constraints_sense == '<'] *= -1

            soc_vars_mask = np.array(['soc' in v.getAttr(gurobipy.GRB.Attr.VarName) for v in model.getVars()])

            # leave SoC-related columns to the end
            A = np.concatenate([A[:,~soc_vars_mask], A[:,soc_vars_mask]], -1)
            c = np.concatenate([c[~soc_vars_mask], c[soc_vars_mask]], -1)

            A_ineq = A[constraints_sense != '=']
            b_ineq = b[constraints_sense != '=']
            A_eq = A[constraints_sense == '=']
            b_eq = b[constraints_sense == '=']

            self.model_data[r_i] = (
                torch.Tensor(A_ineq).to(self.device).double(),
                torch.Tensor(A_eq).to(self.device).double(),
                torch.Tensor(b_ineq).to(self.device).double(),
                torch.Tensor(b_eq).to(self.device).double(),
                torch.Tensor(c).to(self.device).double()
            )

            self.lambdak[r_i] = self.lambda0 * torch.ones(A.shape[0]).to(self.device)

        # leave last job for testing
        train_sampler = SubsetRandomSampler(torch.arange(self.samples_per_problem - 1))
        test_sampler = SubsetRandomSampler(torch.arange(self.samples_per_problem - 1, self.samples_per_problem))

        self.data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=train_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )
        self.val_data = dgl.dataloading.GraphDataLoader(
            data,
            sampler=test_sampler,
            batch_size=self.batch_size,
            drop_last=False
        )

    def get_loss_and_metrics(self, x_hat, r, validation=False):
        start = time()

        x = torch.sigmoid(x_hat)

        As_ineq = list()
        As_eq = list()
        bs_ineq = list()
        bs_eq = list()
        cs = list()
        batch_lambdak = list()
        for r_ in r:
            i = tuple(r_.tolist())

            # get model data
            A_ineq, A_eq, b_ineq, b_eq, c = self.model_data[i]
            As_ineq.append(A_ineq)
            As_eq.append(A_eq)
            bs_ineq.append(b_ineq)
            bs_eq.append(b_eq)
            cs.append(c)

            # get Lag. multipliers estimate for the batch
            lk = self.lambdak[i]
            if lk is None:
                self.lambdak[i] = self.lambda0 * torch.ones(A_ineq.shape[0] + A_eq.shape[0])
                lk = self.lambdak[i]

            batch_lambdak.append(lk)
        batch_lambdak = torch.stack(batch_lambdak)

        batch_A_ineq = torch.stack(As_ineq)
        batch_A_eq = torch.stack(As_eq)
        batch_b_ineq = torch.stack(bs_ineq)
        batch_b_eq = torch.stack(bs_eq)
        batch_c = torch.stack(cs)

        soc = get_soc(x, self.instance, r)
        soc = soc[:,1:]  # discard initial SoC

        x_with_soc = torch.hstack((x, soc))

        C_ineq = torch.bmm(batch_A_ineq, x_with_soc.unsqueeze(-1)).squeeze(-1) - batch_b_ineq
        C_eq = torch.bmm(batch_A_eq, x_with_soc.unsqueeze(-1)).squeeze(-1) - batch_b_eq

        batch_lambdak_ineq = batch_lambdak[:,:C_ineq.shape[1]]
        batch_lambdak_eq = batch_lambdak[:,C_ineq.shape[1]:]

        if (self._e + 1) % 51 == 0:
            # update Lag. multipliers estimate
            with torch.no_grad():
                batch_lambdak_ineq = torch.max(batch_lambdak_ineq - C_ineq / self.muk,
                                               torch.zeros_like(C_ineq))
                batch_lambdak_eq = batch_lambdak_eq - C_eq / self.muk

                batch_lambdak = torch.hstack((batch_lambdak_ineq, batch_lambdak_eq))
                for j, r_ in enumerate(r):
                    i = tuple(r_.tolist())
                    self.lambdak[i] = batch_lambdak[j]

        # TODO: try implicit s through \psi formula
        s_ineq = torch.max(C_ineq - self.muk * batch_lambdak_ineq,
                           torch.zeros_like(C_ineq))

        # compute the augmented lagrangian following Nocedal and Wright
        obj = (x_with_soc * batch_c).sum(-1)
        aug_lagragian = -obj  # cost = - objective (maximization problem)
        # aug_lagragian = 0  # ignore cost for now
        aug_lagragian += (1 / (2 * self.muk))*(C_ineq - s_ineq).pow(2).sum(-1)
        aug_lagragian += -(batch_lambdak_ineq * (C_ineq - s_ineq)).sum(-1)
        aug_lagragian += (1 / (2 * self.muk))*C_eq.pow(2).sum(-1)
        aug_lagragian += -(batch_lambdak_eq * C_eq).sum(-1)
        aug_lagragian = aug_lagragian.mean()  # batch aggregation

        loss_time = time() - start
        if validation:
            curr_muk = self.muk

            if (self._e + 1) % 51 == 0:
                # update muk after every epoch
                with torch.no_grad():
                    C = torch.hstack((C_ineq - s_ineq, C_eq))
                if C.norm(2).item() <= 0.9 * self.best_cons_for_update:
                    self.best_cons_for_update = C.norm(2).item()
                else:
                    self.muk *= .9
                # reset optimizer
                self._load_optim()

            # here you can compute validation-specific metrics
            with torch.no_grad():
                C = torch.hstack((C_ineq - s_ineq, C_eq))
                C = C.norm(2).item()
            return loss_time, aug_lagragian, (obj, C, C_ineq, C_eq, curr_muk)
        else:
            return loss_time, aug_lagragian

    def aggregate_loss_and_metrics(self, loss, size, metrics=None):
        # scale to data size
        loss = loss / size

        losses = {
            'all': loss
            # here you can aggregate metrics computed on the validation set and
            # track them on wandb
        }

        if metrics is not None:
            obj, C, C_ineq, C_eq, muk = metrics[0]
            # TODO: compute distance from integrality
            losses['obj'] = obj.item()
            losses['C'] = C
            losses['mu_k'] = muk
            losses['best_C'] = self.best_cons_for_update
            losses['valid_ineq_ratio'] = (C_ineq >= 0.).sum() / C_ineq.shape[-1]
            losses['valid_eq_ratio'] = (C_eq >= 0.).sum() / C_eq.shape[-1]

        return losses
