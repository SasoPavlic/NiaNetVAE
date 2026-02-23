from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from tabulate import tabulate

from torch import Tensor

from log import Log
from nianetvae.experiments.anomaly_evaluation import *
from nianetvae.experiments.metrics_evaluation import EvaluationMetrics
from nianetvae.experiments.anomaly_evaluation import AnomalyDetectionMetrics

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
                    Log.debug(f"Learning rate: {pl_module.learning_rate}")

                self.previous_loss = pl_module.train_loss['loss'].item()

            else:
                self.lr_find(trainer, pl_module)
                Log.debug(f"Learning rate: {pl_module.learning_rate}")


class RNNVAExperiment(LightningModule):
    def __init__(self, model: BaseVAE, dataset_name, alg_name, **kwargs) -> None:
        super(RNNVAExperiment, self).__init__()

        self.results = None
        self.model = model
        self.dataset_name=dataset_name
        self.alg_name=alg_name
        self.learning_rate = 0.01
        self.params = kwargs['exp_params']
        self.seq_len = kwargs['data_params']['seq_len']
        self.n_features = kwargs['data_params']['n_features']
        self.curr_device = None
        self.hold_graph = False
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.metrics = EvaluationMetrics()
        self.compute_anomaly_metrics = bool(kwargs.get('exp_params', {}).get('compute_anomaly_metrics', True))
        self.anomaly_detection_metrics = AnomalyDetectionMetrics() if self.compute_anomaly_metrics else None
        self.anomaly_metrics = {}
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
        self.train_loss = self.model.loss_function(
            self.curr_device,
            **results,
            M_N=self.params['kld_weight'],
            batch_idx=batch_idx
        )

        # Log the main loss with prog_bar=True to display it in the progress bar
        self.log('train_loss', self.train_loss['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return self.train_loss['loss']

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['signal'].device
        self.val_loss = self.model.loss_function(
            self.curr_device,
            **results,
            M_N=self.params['kld_weight'],
            batch_idx=batch_idx
        )

        # Log the validation loss with prog_bar=True
        self.log('val_loss', self.val_loss['loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        torch.cuda.empty_cache()
        return self.val_loss['loss']

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        results = self.forward(batch)

        # Update evaluation metrics
        self.metrics.to(self.curr_device)
        self.test_loss = self.model.loss_function(
            self.curr_device,
            **results,
            M_N=self.params['kld_weight'],
            batch_idx=batch_idx
        )
        self.metrics.update(results['signal'], results['reconstructed'])
        self.results = self.metrics.compute()

        # Log the results
        self.log_dict(
            self.results,
            prog_bar=True, sync_dist=True, on_step=False,
            on_epoch=True, batch_size=batch['signal'].shape[0]
        )

        # TODO Check if the value of metric here is the same as it is in DB
        # Update anomaly detection metrics
        if self.compute_anomaly_metrics and self.anomaly_detection_metrics is not None:
            self.anomaly_detection_metrics.update(
                predictions=results['reconstructed'],
                targets=batch['signal'],
                labels=batch['target'],
                ts_ids=batch.get('ts_id', None)  # Pass ts_id if available
            )
        # TODO When all dataloaders will return ts_id change to:
        # ts_ids = batch['ts_id']

        torch.cuda.empty_cache()
        return self.results

    def on_test_end(self):
        if not self.compute_anomaly_metrics or self.anomaly_detection_metrics is None:
            self.anomaly_metrics = {}
            Log.debug("Anomaly detection metrics are disabled (exp_params.compute_anomaly_metrics=false).")
            return

        # Compute anomaly detection metrics
        self.anomaly_metrics = self.anomaly_detection_metrics.compute()

        # Helper function to safely format metric values
        def safe_format(value):
            return f"{value:.4f}" if value is not None else "N/A"

        # Print metrics using Tabulate
        if self.anomaly_metrics:
            metrics_list = [
                ["Precision", safe_format(self.anomaly_metrics.get('precision'))],
                ["Recall", safe_format(self.anomaly_metrics.get('recall'))],
                ["F1-Score", safe_format(self.anomaly_metrics.get('f1_score'))],
                ["PR AUC Mean", safe_format(self.anomaly_metrics.get('pr_auc_mean'))],
                ["PR AUC Std", safe_format(self.anomaly_metrics.get('pr_auc_std'))],
                ["ROC AUC Mean", safe_format(self.anomaly_metrics.get('roc_auc_mean'))],
                ["ROC AUC Std", safe_format(self.anomaly_metrics.get('roc_auc_std'))],
            ]
            Log.info("\nAnomaly Detection Metrics:")
            Log.info(tabulate(metrics_list, headers=["Metric", "Value"], tablefmt="pretty"))
        else:
            Log.error("Anomaly detection was not performed due to errors.")

