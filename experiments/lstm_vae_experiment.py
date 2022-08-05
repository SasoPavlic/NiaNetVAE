import math
import os
import sys
import torch
import torchmetrics
from sklearn.metrics import mean_squared_error
from torch import optim
from models.types_ import *
from models import BaseVAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from typing import Any
import torch
from torch import Tensor, tensor


class RMSE(torchmetrics.Metric):
    # https: // www.pytorchlightning.ai / blog / torchmetrics - pytorch - metrics - built - to - scale
    def __init__(self, **kwargs: Any, ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return torch.sqrt(self.sum_squared_error / self.n_observations)


class LSTMVAExperiment(pl.LightningModule):
    def __init__(self,
                 lstm_vae_model: BaseVAE,
                 params: dict,
                 n_features: int) -> None:
        super(LSTMVAExperiment, self).__init__()

        self.model = lstm_vae_model
        self.params = params
        self.n_features = n_features
        self.curr_device = None
        self.hold_graph = False
        self.training_RMSE_metric = RMSE()
        self.validation_RMSE_metric = RMSE()
        self.train_RMSE = None
        self.val_RMSE = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device

        results = self.forward(real_signal)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.training_RMSE_metric(results[0], real_signal)
        return train_loss['loss']

    def training_epoch_end(self, outputs):
        self.log('train_MSE_epoch', self.training_RMSE_metric.compute())

    def on_train_end(self) -> None:
        self.train_RMSE = self.training_RMSE_metric.compute()

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device

        results = self.forward(real_signal)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.log('validation_MSE_step', self.validation_RMSE_metric(results[0], real_signal), on_step=True,
                 on_epoch=False)

    def on_validation_end(self) -> None:
        # self.sample_signals()
        self.val_RMSE = self.validation_RMSE_metric.compute()

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

    def sample_signals(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        # TODO implement for time-series instead of images
        vutils.save_image(recons,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=8, )

        try:
            samples = self.model.sample(self.n_features,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=8)
        except Warning:
            pass
