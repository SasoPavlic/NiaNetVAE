from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from tabulate import tabulate

from torch import Tensor

from log import Log
from nianetvae.experiments.anomaly_detection import *
from nianetvae.experiments.metrics_evaluation import EvaluationMetrics
from nianetvae.experiments.anomaly_detection import AnomalyDetectionMetrics

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
                    Log.info(f"Learning rate: {pl_module.learning_rate}")

                self.previous_loss = pl_module.train_loss['loss'].item()

            else:
                self.lr_find(trainer, pl_module)
                Log.info(f"Learning rate: {pl_module.learning_rate}")


class RNNVAExperiment(LightningModule):
    def __init__(self, model: BaseVAE, path,dataset_name, alg_name, **kwargs) -> None:
        super(RNNVAExperiment, self).__init__()

        self.results = None
        self.model = model
        self.path = path
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
        self.metrics = EvaluationMetrics((self.seq_len * kwargs['data_params']['n_features']))
        self.anomaly_detection_metrics = AnomalyDetectionMetrics()
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

        # Update anomaly detection metrics
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
        # Compute anomaly detection metrics
        save_path = os.path.join(os.getcwd(), self.path)
        self.anomaly_metrics = self.anomaly_detection_metrics.compute(save_path=save_path)

        # Print metrics using Tabulate
        if self.anomaly_metrics:
            metrics_list = [
                ["Precision", f"{self.anomaly_metrics['precision']:.3f}"],
                ["Recall", f"{self.anomaly_metrics['recall']:.3f}"],
                ["F1-Score", f"{self.anomaly_metrics['f1_score']:.3f}"],
                ["PR AUC",
                 f"{self.anomaly_metrics['pr_auc']:.3f}" if self.anomaly_metrics['pr_auc'] is not None else "N/A"],
                ["PR AUC Mean", f"{self.anomaly_metrics['pr_auc_mean']:.3f}" if self.anomaly_metrics[
                                                                                    'pr_auc_mean'] is not None else "N/A"],
                ["PR AUC Std", f"{self.anomaly_metrics['pr_auc_std']:.3f}" if self.anomaly_metrics[
                                                                                  'pr_auc_std'] is not None else "N/A"],
                ["ROC AUC",
                 f"{self.anomaly_metrics['roc_auc']:.3f}" if self.anomaly_metrics['roc_auc'] is not None else "N/A"],

                ["ROC AUC Mean", f"{self.anomaly_metrics['roc_auc_mean']:.3f}" if self.anomaly_metrics[
                                                                                      'roc_auc_mean'] is not None else "N/A"],
                ["ROC AUC Std", f"{self.anomaly_metrics['roc_auc_std']:.3f}" if self.anomaly_metrics[
                                                                                    'roc_auc_std'] is not None else "N/A"],
            ]
            Log.info("\nAnomaly Detection Metrics:")
            Log.info(tabulate(metrics_list, headers=["Metric", "Value"], tablefmt="pretty"))
        else:
            Log.error("Anomaly detection was not performed due to errors.")
