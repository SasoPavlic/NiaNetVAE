import torch
import numpy as np
from torchmetrics.functional import (
    auroc,
    roc,
    precision_recall_curve,
    average_precision,
)
from sklearn.metrics import confusion_matrix
from log import Log
# If you had plotting imports before, keep them disabled to avoid side-effects:
# from nianetvae.experiments.visualization import plot_roc_curve, plot_precision_recall_curve


class AnomalyDetectionMetrics:
    """
    CARLA-style evaluation:
      • For each time series (TS): compute PR/ROC curves on anomaly scores.
      • Pick per-TS optimal threshold (max F1 from PR curve).
      • Convert scores→preds, build per-TS confusion matrix (TP,FP,TN,FN).
      • Sum TP/FP/TN/FN across all TS → dataset-level confusion matrix.
      • Compute dataset-level Precision, Recall, F1 from aggregated counts.
      • Report AU-PR as mean ± std across TS (and AUROC mean ± std optionally).

    Notes:
      • No global thresholding.
      • No averaging of per-TS F1 (non-additive) — follows CARLA §4.4 exactly.
    """

    def __init__(self):
        # Per-series accumulators
        self.errors_per_ts = {}  # ts_id -> [errors...]
        self.labels_per_ts = {}  # ts_id -> [labels...]
        # (Global concatenations kept only for sanity checks if needed; not used for metrics)
        self.all_errors = []
        self.all_labels = []

    def to(self, device):
        # Data kept on CPU by design
        pass

    def update(self, predictions, targets, labels, ts_ids=None):
        """
        Args:
            predictions (Tensor): [batch, seq_len, n_features]
            targets (Tensor):     [batch, seq_len, n_features]
            labels (Tensor):      [batch]  (0/1)
            ts_ids (Tensor|None): [batch]  (integer series IDs)
        """
        batch_errors = self.compute_reconstruction_errors(predictions, targets)
        batch_labels = labels.detach().cpu()

        # Keep global for checks (not used in final metrics)
        self.all_errors.append(batch_errors)
        self.all_labels.append(batch_labels)

        if ts_ids is not None:
            batch_ts_ids = ts_ids.detach().cpu()
            for err, lab, ts in zip(batch_errors, batch_labels, batch_ts_ids):
                ts = int(ts)
                if ts not in self.errors_per_ts:
                    self.errors_per_ts[ts] = []
                    self.labels_per_ts[ts] = []
                self.errors_per_ts[ts].append(err)
                self.labels_per_ts[ts].append(lab)
        else:
            # Treat as a single time series with id=0
            if 0 not in self.errors_per_ts:
                self.errors_per_ts[0] = []
                self.labels_per_ts[0] = []
            self.errors_per_ts[0].extend(batch_errors)
            self.labels_per_ts[0].extend(batch_labels)

    def compute(self):
        """
        Returns:
            dict with:
              precision, recall, f1_score,
              pr_auc_mean, pr_auc_std,
              roc_auc_mean, roc_auc_std
        """
        # Sanity checks
        if len(self.all_errors) == 0 or len(self.all_labels) == 0:
            Log.debug("No data accumulated for metrics computation.")
            return self._empty_metrics()

        all_errors = torch.cat(self.all_errors)
        all_labels = torch.cat(self.all_labels)

        if all_errors.numel() == 0 or all_labels.numel() == 0:
            Log.debug("Empty tensors for errors/labels.")
            return self._empty_metrics()

        if torch.isnan(all_errors).any():
            Log.debug("NaNs in errors; skipping metric computation.")
            return self._empty_metrics()

        if len(all_labels.unique()) < 2:
            Log.debug("Only one class in labels; cannot compute metrics.")
            return self._empty_metrics()

        # -------- CARLA-style per-TS evaluation and aggregation --------
        total_TP = total_FP = total_TN = total_FN = 0
        aupr_list = []
        auroc_list = []

        any_ts_used = False

        for ts_id in self.errors_per_ts:
            # Stack the (possibly many mini-batches) for this TS
            errors = torch.stack(self.errors_per_ts[ts_id])
            labels = torch.stack(self.labels_per_ts[ts_id])

            # Skip TS with NaNs or single-class labels
            if torch.isnan(errors).any():
                Log.debug(f"NaNs in errors for TS {ts_id}; skipping.")
                continue
            if len(labels.unique()) < 2:
                Log.debug(f"Single-class labels for TS {ts_id}; skipping.")
                continue

            try:
                # PR curve and AU-PR for this TS
                pr_prec, pr_rec, pr_thresh = precision_recall_curve(errors, labels.int(), task='binary')
                pr_auc = average_precision(errors, labels.int(), task='binary').item()

                # AUROC for this TS (not used for final F1, but useful to log/compare)
                ts_auroc = auroc(errors, labels.int(), task='binary').item()

                # Choose per-TS threshold that maximizes F1 on PR curve
                pr_f1 = 2 * pr_prec * pr_rec / (pr_prec + pr_rec + 1e-10)
                pr_f1_np = pr_f1.numpy()
                pr_thresh_np = pr_thresh.numpy()
                best_idx = int(np.nanargmax(pr_f1_np))
                ts_threshold = float(pr_thresh_np[best_idx])

                # Threshold scores → predictions for this TS
                preds = self.calculate_anomaly_scores(errors, ts_threshold)

                # Build per-TS confusion matrix
                tn, fp, fn, tp = confusion_matrix(labels.int().numpy(), preds.numpy()).ravel()

                # Aggregate
                total_TP += int(tp)
                total_FP += int(fp)
                total_TN += int(tn)
                total_FN += int(fn)

                aupr_list.append(pr_auc)
                auroc_list.append(ts_auroc)
                any_ts_used = True

            except Exception as e:
                Log.error(f"Error computing TS metrics for {ts_id}: {e}")
                continue

        # If no TS was usable, return empty metrics
        if not any_ts_used:
            Log.debug("No valid time series contributed to metrics.")
            return self._empty_metrics()

        # Dataset-level Precision/Recall/F1 from aggregated counts
        precision_agg = None
        recall_agg = None
        f1_agg = None

        denom_p = (total_TP + total_FP)
        denom_r = (total_TP + total_FN)

        if denom_p > 0:
            precision_agg = total_TP / denom_p
        if denom_r > 0:
            recall_agg = total_TP / denom_r
        if precision_agg is not None and recall_agg is not None and (precision_agg + recall_agg) > 0:
            f1_agg = 2.0 * precision_agg * recall_agg / (precision_agg + recall_agg)

        # AU-PR mean ± std across TS (CARLA tables report this)
        pr_auc_mean = float(np.mean(aupr_list)) if len(aupr_list) > 0 else None
        pr_auc_std = float(np.std(aupr_list)) if len(aupr_list) > 0 else None

        # (Optional) AUROC mean ± std across TS
        roc_auc_mean = float(np.mean(auroc_list)) if len(auroc_list) > 0 else None
        roc_auc_std = float(np.std(auroc_list)) if len(auroc_list) > 0 else None

        return {
            "precision": round(precision_agg, 4) if precision_agg is not None else None,
            "recall": round(recall_agg, 4) if recall_agg is not None else None,
            "f1_score": round(f1_agg, 4) if f1_agg is not None else None,
            "pr_auc_mean": round(pr_auc_mean, 4) if pr_auc_mean is not None else None,
            "pr_auc_std": round(pr_auc_std, 4) if pr_auc_std is not None else None,
            "roc_auc_mean": round(roc_auc_mean, 4) if roc_auc_mean is not None else None,
            "roc_auc_std": round(roc_auc_std, 4) if roc_auc_std is not None else None,
        }

    @staticmethod
    def _empty_metrics():
        return {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "pr_auc_mean": None,
            "pr_auc_std": None,
            "roc_auc_mean": None,
            "roc_auc_std": None,
        }

    @staticmethod
    def compute_reconstruction_errors(predictions, targets):
        """
        Mean squared reconstruction error per sample (across seq_len and features).
        Shapes: [batch, seq_len, n_features] -> [batch]
        """
        errors = torch.mean((predictions - targets) ** 2, dim=(1, 2))
        return errors.detach().cpu()

    @staticmethod
    def calculate_anomaly_scores(errors, threshold: float):
        """
        Convert anomaly scores (errors) to binary predictions using given threshold.
        """
        return (errors >= threshold).int()
