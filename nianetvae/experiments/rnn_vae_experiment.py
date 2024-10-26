import os
from typing import Any

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from torch import Tensor, tensor

from log import Log
from nianetvae.experiments.anomaly_detection import *
from nianetvae.experiments.evaluationmetrics import EvaluationMetrics
from nianetvae.experiments.visualization import plot_roc_curve
from nianetvae.models.base import BaseVAE


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs['lr_finder'])
        self.tune_n_epochs = kwargs['tune_n_epochs']
        self.previous_loss = float('inf')

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.tune_n_epochs == 0 or trainer.current_epoch == 0:
            if pl_module.train_loss is not None:
                loss = pl_module.train_loss['loss'].item()
                if loss < self.previous_loss:
                    Log.debug(f"\nLoss decreased from {self.previous_loss} to {loss}")

                elif loss > self.previous_loss:
                    Log.debug(f"\nLoss increased from {self.previous_loss} to {loss}")
                    self.lr_find(trainer, pl_module)
                    print(f"Learning rate: {pl_module.learning_rate}")

                self.previous_loss = pl_module.train_loss['loss'].item()

            else:
                self.lr_find(trainer, pl_module)
                print(f"Learning rate: {pl_module.learning_rate}")


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


class RNNVAExperiment(LightningModule):
    def __init__(self,
                 lstm_vae_model: BaseVAE, path, **kwargs) -> None:
        super(RNNVAExperiment, self).__init__()

        self.results = None
        self.model = lstm_vae_model
        self.path = path
        self.learning_rate = 0.01
        self.params = kwargs['exp_params']
        self.seq_len = kwargs['data_params']['seq_len']
        self.n_features = kwargs['data_params']['n_features']
        self.curr_device = None
        self.hold_graph = False
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.metrics = EvaluationMetrics(num_outputs=(self.seq_len * kwargs['data_params']['n_features']))
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def configure_optimizers(self):

        """When AE does not have any layers"""
        if len(list(self.model.parameters())) == 0:
            self.model.optimizer_name = "Empty"
            return None

        if self.model.optimizer_name == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "Adagrad":
            return torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "RAdam":
            return torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "ASGD":
            return torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "RPROP":
            return torch.optim.Rprop(self.model.parameters(), lr=self.learning_rate)

        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['signal'].device
        self.train_loss = self.model.loss_function(self.curr_device,
                                                 **results,
                                                 M_N=self.params['kld_weight'],
                                                 batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in self.train_loss.items()}, sync_dist=True, on_step=True,
                      on_epoch=True)
        torch.cuda.empty_cache()
        return self.train_loss['loss']

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['signal'].device
        self.val_loss = self.model.loss_function(self.curr_device,
                                                 **results,
                                                 M_N=self.params['kld_weight'],
                                                 batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in self.val_loss.items()}, sync_dist=True, on_step=False,
                      on_epoch=True)
        # TODO add more metrics
        # https://github.com/Lightning-AI/metrics/issues/340#issuecomment-872073730
        torch.cuda.empty_cache()
        return self.val_loss['loss']

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        torch.cuda.empty_cache()
        results = self.forward(batch)

        self.metrics.to(self.curr_device)

        self.test_loss = self.model.loss_function(self.curr_device,
                                                 **results,
                                                 M_N=self.params['kld_weight'],
                                                 batch_idx=batch_idx)

        self.metrics.update(results['signal'], results['reconstructed'])

        self.results = self.metrics.compute()

        self.log_dict(self.results,
                      prog_bar=True, sync_dist=True, on_step=False,
                      on_epoch=True, batch_size=batch['signal'].shape[0])

        torch.cuda.empty_cache()

    def on_train_end(self):
        # Perform anomaly detection after training is complete

        all_predictions = []
        all_targets = []
        all_labels = []

        dataloader = self.trainer.datamodule.test_dataloader()

        with torch.no_grad():
            for batch in dataloader:
                batch['signal'] = batch['signal'].to(self.device)
                batch['target'] = batch['target'].to(self.device)

                x = batch['signal']  # x now has shape (batch_size, seq_len, n_features)
                labels = batch['target']

                response = self.model(batch)  # Pass x directly to the model

                all_predictions.append(response["reconstructed"])
                all_targets.append(x)
                all_labels.append(labels)

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Perform anomaly detection
        save_path = os.path.join(os.getcwd(), self.path)
        self.anomaly_metrics = perform_anomaly_detection(
            all_predictions, all_targets, all_labels, save_path=save_path
        )

        if self.anomaly_metrics:
            # Print metrics for debugging
            print("Anomaly Detection Metrics:")
            print(f"Precision: {self.anomaly_metrics['precision']}")
            print(f"Recall: {self.anomaly_metrics['recall']}")
            print(f"F1-Score: {self.anomaly_metrics['f1_score']}")
            print(f"ROC AUC: {self.anomaly_metrics['roc_auc']}")
        else:
            print("Anomaly detection was not performed due to errors.")
