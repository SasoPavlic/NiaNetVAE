from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from log import Log


class WindowAnomalyRankingMetrics:
    """
    MetroPT-aligned window-ranking metrics.

    Each test window receives one anomaly score: mean squared reconstruction error
    over sequence length and features. Metrics are computed globally across all test
    windows in the current cycle, not averaged per segment/time-series.
    """

    def __init__(self):
        self.all_scores = []
        self.all_labels = []
        self.all_ts_ids = []

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
            diagnostics.update(self._invalid_ranking_payload("non_finite_scores"))
            return diagnostics

        positive_count = int(np.sum(labels == 1))
        negative_count = int(np.sum(labels == 0))
        if positive_count <= 0:
            diagnostics.update(self._invalid_ranking_payload("no_positive_windows"))
            return diagnostics
        if negative_count <= 0:
            diagnostics.update(self._invalid_ranking_payload("no_negative_windows"))
            return diagnostics

        try:
            window_auprc = float(average_precision_score(labels, scores))
            window_roc_auc = float(roc_auc_score(labels, scores))
            best_f1 = self._best_f1_diagnostics(labels=labels, scores=scores)
        except Exception as exc:
            Log.error(f"Error computing window anomaly ranking metrics: {exc}")
            diagnostics.update(self._invalid_ranking_payload(f"ranking_metric_failed:{exc.__class__.__name__}"))
            return diagnostics

        diagnostics.update(
            {
                "window_auprc": round(window_auprc, 4),
                "window_roc_auc": round(window_roc_auc, 4),
                "ranking_metric_valid": True,
                "ranking_metric_invalid_reason": None,
                **best_f1,
            }
        )
        return diagnostics

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
            "score_min": round(float(np.min(finite_scores)), 6) if finite_scores.size else None,
            "score_max": round(float(np.max(finite_scores)), 6) if finite_scores.size else None,
            "score_mean": round(float(np.mean(finite_scores)), 6) if finite_scores.size else None,
            "score_std": round(float(np.std(finite_scores)), 6) if finite_scores.size else None,
            "segment_count": int(np.unique(ts_ids).shape[0]) if ts_ids.size else 0,
        }

    @staticmethod
    def _best_f1_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        if thresholds.size <= 0:
            return {
                "best_f1_threshold": None,
                "best_f1_precision": None,
                "best_f1_recall": None,
                "best_f1_score": None,
            }

        # The last PR point has no corresponding threshold; exclude it for thresholded diagnostics.
        threshold_precision = precision[:-1]
        threshold_recall = recall[:-1]
        f1 = (
            2.0
            * threshold_precision
            * threshold_recall
            / (threshold_precision + threshold_recall + 1e-12)
        )
        if f1.size <= 0 or not np.isfinite(f1).any():
            return {
                "best_f1_threshold": None,
                "best_f1_precision": None,
                "best_f1_recall": None,
                "best_f1_score": None,
            }
        best_idx = int(np.nanargmax(f1))
        return {
            "best_f1_threshold": round(float(thresholds[best_idx]), 6),
            "best_f1_precision": round(float(threshold_precision[best_idx]), 4),
            "best_f1_recall": round(float(threshold_recall[best_idx]), 4),
            "best_f1_score": round(float(f1[best_idx]), 4),
        }

    @classmethod
    def _empty_metrics(cls, reason: str) -> dict:
        payload = cls._base_empty_payload()
        payload.update(cls._invalid_ranking_payload(reason))
        return payload

    @staticmethod
    def _base_empty_payload() -> dict:
        return {
            "window_count": 0,
            "positive_window_count": 0,
            "negative_window_count": 0,
            "positive_window_rate": None,
            "score_min": None,
            "score_max": None,
            "score_mean": None,
            "score_std": None,
            "segment_count": 0,
        }

    @staticmethod
    def _invalid_ranking_payload(reason: str) -> dict:
        return {
            "window_auprc": None,
            "window_roc_auc": None,
            "ranking_metric_valid": False,
            "ranking_metric_invalid_reason": reason,
            "best_f1_threshold": None,
            "best_f1_precision": None,
            "best_f1_recall": None,
            "best_f1_score": None,
        }

    @staticmethod
    def compute_reconstruction_errors(predictions, targets):
        """
        Mean squared reconstruction error per window.
        Shapes: [batch, seq_len, n_features] -> [batch]
        """
        errors = torch.mean((predictions - targets) ** 2, dim=(1, 2))
        return errors.detach().cpu()

    @staticmethod
    def calculate_anomaly_scores(scores, threshold: float):
        """Convert anomaly scores to binary predictions using a threshold."""
        return (scores >= threshold).int()
