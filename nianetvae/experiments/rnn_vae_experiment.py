from lightning.pytorch import LightningModule
from tabulate import tabulate

import torch
from torch import Tensor

from log import Log
from nianetvae.experiments.metrics_evaluation import EvaluationMetrics
from nianetvae.experiments.anomaly_evaluation import WindowAnomalyRankingMetrics

from nianetvae.models.base import BaseVAE


class RNNVAExperiment(LightningModule):
    def __init__(self, model: BaseVAE, dataset_name, alg_name, **kwargs) -> None:
        super(RNNVAExperiment, self).__init__()

        self.results = None
        self.model = model
        self.dataset_name=dataset_name
        self.alg_name=alg_name
        self.params = kwargs['exp_params']
        self.learning_rate = float(self.params.get('learning_rate', 0.003))
        self.weight_decay = float(self.params.get('weight_decay', 0.0))
        self.seq_len = kwargs['data_params']['seq_len']
        self.n_features = kwargs['data_params']['n_features']
        self.curr_device = None
        self.hold_graph = False
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.metrics = EvaluationMetrics()
        # C13.3 policy lock: anomaly metrics are always enabled because obj_pdm depends on them.
        self.compute_anomaly_metrics = True
        self.anomaly_detection_metrics = WindowAnomalyRankingMetrics()
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

        if self.model.optimizer_name != "Adam":
            raise ValueError(
                f"Unsupported optimizer {self.model.optimizer_name!r}. "
                "Architecture-only search requires fixed Adam training."
            )

        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

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
        self.anomaly_metrics = self.anomaly_detection_metrics.compute()

        # Helper function to safely format metric values
        def safe_format(value):
            return f"{value:.4f}" if value is not None else "N/A"

        # Print metrics using Tabulate
        if self.anomaly_metrics:
            metrics_list = [
                ["Window AUPRC", safe_format(self.anomaly_metrics.get('window_auprc'))],
                ["Window ROC AUC", safe_format(self.anomaly_metrics.get('window_roc_auc'))],
                ["Ranking Valid", str(self.anomaly_metrics.get('ranking_metric_valid'))],
                ["Invalid Reason", self.anomaly_metrics.get('ranking_metric_invalid_reason') or "N/A"],
                ["Window Count", self.anomaly_metrics.get('window_count')],
                ["Positive Windows", self.anomaly_metrics.get('positive_window_count')],
                ["Negative Windows", self.anomaly_metrics.get('negative_window_count')],
                ["Positive Rate", safe_format(self.anomaly_metrics.get('positive_window_rate'))],
                ["Score Min", safe_format(self.anomaly_metrics.get('score_min'))],
                ["Score Max", safe_format(self.anomaly_metrics.get('score_max'))],
                ["Score Mean", safe_format(self.anomaly_metrics.get('score_mean'))],
                ["Score Std", safe_format(self.anomaly_metrics.get('score_std'))],
                ["Segment Count", self.anomaly_metrics.get('segment_count')],
                ["Best-F1 Threshold", safe_format(self.anomaly_metrics.get('best_f1_threshold'))],
                ["Best-F1 Precision", safe_format(self.anomaly_metrics.get('best_f1_precision'))],
                ["Best-F1 Recall", safe_format(self.anomaly_metrics.get('best_f1_recall'))],
                ["Best-F1 Score", safe_format(self.anomaly_metrics.get('best_f1_score'))],
            ]
            Log.info("\nWindow Anomaly Ranking Metrics:")
            Log.info(tabulate(metrics_list, headers=["Metric", "Value"], tablefmt="pretty"))
        else:
            Log.error("Anomaly detection was not performed due to errors.")
