from __future__ import annotations

import numpy as np
import torch

from log import Log


class WindowAnomalyRankingMetrics:
    """
    MetroPT-aligned calibrated PdM objective metrics.

    Each test window receives one window reconstruction error score:
    mean squared reconstruction error
    over sequence length and features. Raw reconstruction errors are converted to
    percentile risk scores against the training-window calibration distribution.
    """

    def __init__(self):
        self.all_scores = []
        self.all_labels = []
        self.all_ts_ids = []
        self.calibration_scores = []

    def to(self, device):
        # Data is accumulated on CPU by design.
        return self

    def update(self, predictions, targets, labels, ts_ids=None):
        """
        Args:
            predictions (Tensor): [batch, seq_len, n_features]
            targets (Tensor):     [batch, seq_len, n_features]
            labels (Tensor):      [batch] binary end-anchor phase labels
            ts_ids (Tensor|None): [batch] segment IDs; used only for diagnostics
        """
        batch_scores = self.compute_reconstruction_errors(predictions, targets)
        batch_labels = labels.detach().cpu().reshape(-1).int()
        if ts_ids is None:
            batch_ts_ids = torch.zeros_like(batch_labels, dtype=torch.int64)
        else:
            batch_ts_ids = ts_ids.detach().cpu().reshape(-1).long()

        self.all_scores.append(batch_scores.reshape(-1))
        self.all_labels.append(batch_labels)
        self.all_ts_ids.append(batch_ts_ids)

    def update_calibration(self, predictions, targets):
        """
        Accumulate final-model reconstruction errors on training windows.

        These scores define the calibration distribution used to transform raw
        reconstruction errors into MetroPT-style percentile risk_score values.
        """
        batch_scores = self.compute_reconstruction_errors(predictions, targets)
        self.calibration_scores.append(batch_scores.reshape(-1))

    def compute(self):
        if len(self.all_scores) == 0 or len(self.all_labels) == 0:
            Log.debug("No data accumulated for window anomaly ranking metrics.")
            return self._empty_metrics("no_windows")

        scores = torch.cat(self.all_scores).detach().cpu().numpy().astype(np.float64, copy=False)
        labels = torch.cat(self.all_labels).detach().cpu().numpy().astype(np.int64, copy=False)
        ts_ids = torch.cat(self.all_ts_ids).detach().cpu().numpy().astype(np.int64, copy=False)

        if scores.size == 0 or labels.size == 0:
            return self._empty_metrics("no_windows")
        if scores.shape[0] != labels.shape[0]:
            return self._empty_metrics("score_label_length_mismatch")

        diagnostics = self._diagnostics(scores=scores, labels=labels, ts_ids=ts_ids)

        if not np.isfinite(scores).all():
            reason = "non_finite_scores"
            diagnostics.update(self._invalid_pdm_payload(reason))
            return diagnostics

        positive_count = int(np.sum(labels == 1))
        negative_count = int(np.sum(labels == 0))
        if positive_count <= 0:
            reason = "no_positive_windows"
            diagnostics.update(self._invalid_pdm_payload(reason))
            return diagnostics
        if negative_count <= 0:
            reason = "no_negative_windows"
            diagnostics.update(self._invalid_pdm_payload(reason))
            return diagnostics

        try:
            pdm_diag = self._calibrated_risk_gap_diagnostics(labels=labels, scores=scores)
        except Exception as exc:
            Log.error(f"Error computing calibrated PdM metrics: {exc}")
            reason = f"pdm_metric_failed:{exc.__class__.__name__}"
            diagnostics.update(self._invalid_pdm_payload(reason))
            return diagnostics

        diagnostics.update(pdm_diag)
        return diagnostics

    def _calibrated_risk_gap_diagnostics(self, labels: np.ndarray, scores: np.ndarray) -> dict:
        calibration_scores = self._calibration_scores_array()
        if calibration_scores.size <= 0:
            reason = "missing_calibration_scores"
            return self._invalid_pdm_payload(reason)
        if not np.isfinite(calibration_scores).all():
            reason = "non_finite_calibration_scores"
            return self._invalid_pdm_payload(reason)

        sorted_calibration = np.sort(calibration_scores.astype(np.float64, copy=False))
        ranks = np.searchsorted(sorted_calibration, scores, side="right")
        risk_scores = np.clip(ranks / float(sorted_calibration.size), 0.0, 1.0)
        positive_risk_scores = risk_scores[labels == 1]
        negative_risk_scores = risk_scores[labels == 0]
        positive_risk_mean = float(np.mean(positive_risk_scores))
        negative_risk_mean = float(np.mean(negative_risk_scores))
        risk_gap = float(positive_risk_mean - negative_risk_mean)

        return {
            "calibration_window_count": int(calibration_scores.shape[0]),
            "calibration_window_reconstruction_error_min": round(float(np.min(calibration_scores)), 6),
            "calibration_window_reconstruction_error_max": round(float(np.max(calibration_scores)), 6),
            "calibration_window_reconstruction_error_mean": round(float(np.mean(calibration_scores)), 6),
            "calibration_window_reconstruction_error_std": round(float(np.std(calibration_scores)), 6),
            "risk_score_min": round(float(np.min(risk_scores)), 6),
            "risk_score_max": round(float(np.max(risk_scores)), 6),
            "risk_score_mean": round(float(np.mean(risk_scores)), 6),
            "risk_score_std": round(float(np.std(risk_scores)), 6),
            "pdm_positive_risk_mean": round(positive_risk_mean, 6),
            "pdm_negative_risk_mean": round(negative_risk_mean, 6),
            "pdm_risk_gap": round(risk_gap, 6),
            "pdm_metric_valid": True,
            "pdm_metric_invalid_reason": None,
        }

    def _calibration_scores_array(self) -> np.ndarray:
        if not self.calibration_scores:
            return np.asarray([], dtype=np.float64)
        return torch.cat(self.calibration_scores).detach().cpu().numpy().astype(np.float64, copy=False)

    @staticmethod
    def _diagnostics(scores: np.ndarray, labels: np.ndarray, ts_ids: np.ndarray) -> dict:
        finite_scores = scores[np.isfinite(scores)]
        window_count = int(scores.shape[0])
        positive_count = int(np.sum(labels == 1))
        negative_count = int(np.sum(labels == 0))
        positive_rate = float(positive_count / window_count) if window_count > 0 else None
        return {
            "window_count": window_count,
            "positive_window_count": positive_count,
            "negative_window_count": negative_count,
            "positive_window_rate": round(positive_rate, 6) if positive_rate is not None else None,
            "window_reconstruction_error_min": round(float(np.min(finite_scores)), 6) if finite_scores.size else None,
            "window_reconstruction_error_max": round(float(np.max(finite_scores)), 6) if finite_scores.size else None,
            "window_reconstruction_error_mean": round(float(np.mean(finite_scores)), 6) if finite_scores.size else None,
            "window_reconstruction_error_std": round(float(np.std(finite_scores)), 6) if finite_scores.size else None,
            "segment_count": int(np.unique(ts_ids).shape[0]) if ts_ids.size else 0,
        }

    def _empty_metrics(self, reason: str) -> dict:
        payload = self._base_empty_payload()
        payload.update(self._invalid_pdm_payload(reason))
        return payload

    @staticmethod
    def _base_empty_payload() -> dict:
        return {
            "window_count": 0,
            "positive_window_count": 0,
            "negative_window_count": 0,
            "positive_window_rate": None,
            "window_reconstruction_error_min": None,
            "window_reconstruction_error_max": None,
            "window_reconstruction_error_mean": None,
            "window_reconstruction_error_std": None,
            "segment_count": 0,
        }

    def _invalid_pdm_payload(self, reason: str) -> dict:
        return {
            "calibration_window_count": 0,
            "calibration_window_reconstruction_error_min": None,
            "calibration_window_reconstruction_error_max": None,
            "calibration_window_reconstruction_error_mean": None,
            "calibration_window_reconstruction_error_std": None,
            "risk_score_min": None,
            "risk_score_max": None,
            "risk_score_mean": None,
            "risk_score_std": None,
            "pdm_positive_risk_mean": None,
            "pdm_negative_risk_mean": None,
            "pdm_risk_gap": None,
            "pdm_metric_valid": False,
            "pdm_metric_invalid_reason": reason,
        }

    @staticmethod
    def compute_reconstruction_errors(predictions, targets):
        """
        Mean squared reconstruction error per window.
        Shapes: [batch, seq_len, n_features] -> [batch]
        """
        errors = torch.mean((predictions - targets) ** 2, dim=(1, 2))
        return errors.detach().cpu()
