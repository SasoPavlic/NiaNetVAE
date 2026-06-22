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
        objectives_cfg = kwargs.get("objectives") or {}
        pdm_cfg = objectives_cfg.get("pdm") or {}
        alarm_burden_cfg = objectives_cfg.get("alarm_burden") or {}
        smoothing_window_windows = int(pdm_cfg.get("smoothing_window_windows", 480))
        alarm_burden_threshold = float(alarm_burden_cfg.get("risk_threshold", 0.95))
        self.anomaly_detection_metrics = WindowAnomalyRankingMetrics(
            smoothing_window_windows=smoothing_window_windows,
            alarm_burden_threshold=alarm_burden_threshold,
        )
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

    def collect_calibration_scores(self, dataloader) -> None:
        """Score final-model training windows for risk-score calibration."""
        if dataloader is None:
            return
        was_training = bool(self.training)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(device)
                results = self.forward({"signal": signal})
                self.anomaly_detection_metrics.update_calibration(
                    predictions=results["reconstructed"],
                    targets=results["signal"],
                )
        if was_training:
            self.train()

    def on_test_end(self):
        # Compute anomaly detection metrics
        self.anomaly_metrics = self.anomaly_detection_metrics.compute()

        # Helper function to safely format metric values
        def safe_format(value):
            return f"{value:.4f}" if value is not None else "N/A"

        # Print metrics using Tabulate
        if self.anomaly_metrics:
            metrics_list = [
                ["Window Count", self.anomaly_metrics.get('window_count')],
                ["Positive Windows", self.anomaly_metrics.get('positive_window_count')],
                ["Negative Windows", self.anomaly_metrics.get('negative_window_count')],
                ["Positive Rate", safe_format(self.anomaly_metrics.get('positive_window_rate'))],
                ["Window Reconstruction Error Min", safe_format(self.anomaly_metrics.get('window_reconstruction_error_min'))],
                ["Window Reconstruction Error Max", safe_format(self.anomaly_metrics.get('window_reconstruction_error_max'))],
                ["Window Reconstruction Error Mean", safe_format(self.anomaly_metrics.get('window_reconstruction_error_mean'))],
                ["Window Reconstruction Error Std", safe_format(self.anomaly_metrics.get('window_reconstruction_error_std'))],
                ["Segment Count", self.anomaly_metrics.get('segment_count')],
                ["Calibration Windows", self.anomaly_metrics.get('calibration_window_count')],
                ["Risk Score Min", safe_format(self.anomaly_metrics.get('risk_score_min'))],
                ["Risk Score Max", safe_format(self.anomaly_metrics.get('risk_score_max'))],
                ["Risk Score Mean", safe_format(self.anomaly_metrics.get('risk_score_mean'))],
                ["PdM Smoothing Window", self.anomaly_metrics.get('pdm_smoothing_window_windows')],
                ["PdM Alarm Burden Threshold", safe_format(self.anomaly_metrics.get('pdm_alarm_burden_threshold'))],
                ["PdM Positive Smoothed Risk Mean", safe_format(self.anomaly_metrics.get('pdm_positive_smoothed_risk_mean'))],
                ["PdM Negative Smoothed Risk Mean", safe_format(self.anomaly_metrics.get('pdm_negative_smoothed_risk_mean'))],
                ["PdM Positive High Risk Rate", safe_format(self.anomaly_metrics.get('pdm_positive_high_risk_rate'))],
                ["PdM Negative High Risk Rate", safe_format(self.anomaly_metrics.get('pdm_negative_high_risk_rate'))],
                ["PdM Smoothed AUROC", safe_format(self.anomaly_metrics.get('pdm_smoothed_auroc'))],
                ["PdM Smoothed Rank Gap", safe_format(self.anomaly_metrics.get('pdm_smoothed_rank_gap'))],
                ["PdM Metric Valid", str(self.anomaly_metrics.get('pdm_metric_valid'))],
                ["PdM Invalid Reason", self.anomaly_metrics.get('pdm_metric_invalid_reason') or "N/A"],
            ]
            Log.info("\nCalibrated PdM Objective Metrics:")
            Log.info(tabulate(metrics_list, headers=["Metric", "Value"], tablefmt="pretty"))
        else:
            Log.error("Anomaly detection was not performed due to errors.")
