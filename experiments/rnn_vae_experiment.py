import math
import os
import statistics
import sys
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from sklearn.metrics import mean_squared_error
from torch import optim
from models.types_ import *
from models import BaseVAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from typing import Any
import torch
from torch import Tensor, tensor


class RNNVAExperiment(LightningModule):
    def __init__(self,
                 lstm_vae_model: BaseVAE,
                 params: dict,
                 n_features: int) -> None:
        super(RNNVAExperiment, self).__init__()

        self.model = lstm_vae_model
        self.params = params
        self.n_features = n_features
        self.curr_device = None
        self.hold_graph = False
        self.test_RMSE = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = self.model.optimizer
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device

        results = self.forward(real_signal)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device

        results = self.forward(real_signal)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        # TODO add more metrics
        # https: // github.com / Lightning - AI / metrics / issues / 340  # issuecomment-872073730

    def on_fit_end(self) -> None:
        self.test_model()

    def test_model(self):

        dataloader_iterator = iter(self.trainer.datamodule.test_dataloader())

        while True:
            try:
                data, target = next(dataloader_iterator)
            except StopIteration:
                break
            finally:
                self.model.testing_RMSE_metric.to('cuda')
                recons = self.model.generate(data, labels=target)
                self.model.testing_RMSE_metric.update(recons, data)

        self.test_RMSE = self.model.testing_RMSE_metric.compute()

    def sample_signals(self):
        try:
            samples = self.model.sample(self.n_features,
                                        self.curr_device)
            pass

        except Warning:
            pass
