import os
import torch
import numpy as np
from torchmetrics.functional import (
    auroc,
    roc,
    precision_recall_curve,
    average_precision,
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from log import Log
from nianetvae.experiments.visualization import plot_roc_curve, plot_precision_recall_curve


class AnomalyDetectionMetrics:
    def __init__(self):
        # Initialize dictionaries to store errors and labels per time series
        self.errors_per_ts = {}
        self.labels_per_ts = {}
        # Initialize lists to store all errors and labels
        self.all_errors = []
        self.all_labels = []

    def to(self, device):
        pass  # Data is stored on CPU

    def update(self, predictions, targets, labels, ts_ids=None):
        """
        Update internal state with new batch data.

        Args:
            predictions (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_features].
            targets (torch.Tensor): True targets of shape [batch_size, seq_len, n_features].
            labels (torch.Tensor): True labels of shape [batch_size].
            ts_ids (torch.Tensor or None): Time series IDs of shape [batch_size], if available.
        """
        # Compute reconstruction errors for the batch
        batch_errors = self.compute_reconstruction_errors(predictions, targets)
        # Ensure labels are on CPU
        batch_labels = labels.detach().cpu()

        # Append to all errors and labels
        self.all_errors.append(batch_errors)
        self.all_labels.append(batch_labels)

        if ts_ids is not None:
            batch_ts_ids = ts_ids.detach().cpu()
            # Update errors and labels per time series
            for error, label, ts_id in zip(batch_errors, batch_labels, batch_ts_ids):
                ts_id = int(ts_id)
                if ts_id not in self.errors_per_ts:
                    self.errors_per_ts[ts_id] = []
                    self.labels_per_ts[ts_id] = []
                self.errors_per_ts[ts_id].append(error)
                self.labels_per_ts[ts_id].append(label)
        else:
            # If ts_ids are not provided, treat all data as belonging to a single time series with ts_id=0
            if 0 not in self.errors_per_ts:
                self.errors_per_ts[0] = []
                self.labels_per_ts[0] = []
            self.errors_per_ts[0].extend(batch_errors)
            self.labels_per_ts[0].extend(batch_labels)

    def compute(self, save_path=None):
        """
        Compute the anomaly detection metrics based on accumulated data.

        Args:
            save_path (str): Directory to save plots, if any.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Concatenate all errors and labels
        all_errors = torch.cat(self.all_errors)
        all_labels = torch.cat(self.all_labels)

        # Check for NaN values in errors
        if torch.isnan(all_errors).any():
            Log.debug("Errors contain NaN values. Skipping anomaly detection for this model.")
            return self._empty_metrics()

        if len(all_labels.unique()) < 2:
            Log.debug("Only one class present in labels. Cannot compute ROC AUC or PR AUC.")
            return self._empty_metrics()

        # Compute overall ROC AUC and PR AUC
        try:
            roc_auc = auroc(all_errors, all_labels.int(), task='binary').item()
            pr_auc = average_precision(all_errors, all_labels.int(), task='binary').item()
        except Exception as e:
            Log.error(f"Error computing overall ROC AUC or PR AUC: {e}")
            roc_auc = pr_auc = None

        # Determine thresholds and calculate overall metrics
        threshold_info = self.determine_thresholds(all_errors, all_labels)
        if threshold_info is None:
            Log.debug("Could not determine thresholds due to errors in ROC or PR curve computation.")
            return self._empty_metrics()

        (
            optimal_threshold,
            fpr, tpr, roc_thresholds, _,
            precision_vals, recall_vals, pr_thresholds, _, _
        ) = threshold_info

        # Classify anomalies based on the optimal threshold from ROC curve
        anomalies = self.calculate_anomaly_scores(all_errors, optimal_threshold)
        precision, recall, f1_score = self.calculate_evaluation_metrics(anomalies, all_labels)

        # Now compute per-time series metrics
        total_TP = total_FP = total_TN = total_FN = 0
        aupr_list = []
        auroc_list = []  # List to store ROC AUC per time series

        for ts_id in self.errors_per_ts:
            errors = torch.stack(self.errors_per_ts[ts_id])
            labels = torch.stack(self.labels_per_ts[ts_id])

            # Compute metrics for this time series
            metrics = self.compute_metrics_per_ts(errors, labels, ts_id, save_path)

            if metrics is None:
                continue  # Skip if metrics could not be computed

            # Update total confusion matrix components
            total_TP += metrics['TP']
            total_FP += metrics['FP']
            total_TN += metrics['TN']
            total_FN += metrics['FN']

            # Collect AU-PR and ROC AUC for averaging
            aupr_list.append(metrics['pr_auc'])
            auroc_list.append(metrics['roc_auc'])  # Collect ROC AUC

        # Calculate average and std of AU-PR
        if aupr_list:
            pr_auc_mean = np.mean(aupr_list)
            pr_auc_std = np.std(aupr_list)
        else:
            pr_auc_mean = pr_auc_std = None

        # Calculate average and std of ROC AUC
        if auroc_list:
            roc_auc_mean = np.mean(auroc_list)
            roc_auc_std = np.std(auroc_list)
        else:
            roc_auc_mean = roc_auc_std = None

        # Return aggregated metrics
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
            'pr_auc': round(pr_auc, 4) if pr_auc is not None else None,
            'pr_auc_mean': round(pr_auc_mean, 4) if pr_auc_mean is not None else None,
            'pr_auc_std': round(pr_auc_std, 4) if pr_auc_std is not None else None,
            'roc_auc_mean': round(roc_auc_mean, 4) if roc_auc_mean is not None else None,
            'roc_auc_std': round(roc_auc_std, 4) if roc_auc_std is not None else None
        }

    @staticmethod
    def _empty_metrics():
        return {
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None,
            'pr_auc': None,
            'pr_auc_mean': None,
            'pr_auc_std': None,
            'roc_auc_mean': None,
            'roc_auc_std': None
        }

    def compute_metrics_per_ts(self, errors, labels, ts_id, save_path=None):
        """
        Compute metrics for a single time series.

        Args:
            errors (torch.Tensor): Reconstruction errors for this time series.
            labels (torch.Tensor): True labels for this time series.
            ts_id (int): Time series ID.
            save_path (str): Directory to save plots, if any.

        Returns:
            dict: Metrics for this time series.
        """
        if torch.isnan(errors).any():
            Log.debug(f"Errors contain NaN values for time series {ts_id}. Skipping.")
            return None

        if len(labels.unique()) < 2:
            Log.debug(f"Only one class present in labels for time series {ts_id}. Skipping.")
            return None

        try:
            # Compute ROC curve and AUC
            fpr, tpr, roc_thresholds = roc(errors, labels.int(), task='binary')
            roc_auc = auroc(errors, labels.int(), task='binary')

            # Compute Precision-Recall curve and AUC
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(errors, labels.int(), task='binary')
            pr_auc = average_precision(errors, labels.int(), task='binary')

            # Determine optimal threshold using F1 score from PR curve
            pr_fscore = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-10)
            pr_fscore = pr_fscore.numpy()
            pr_thresholds_np = pr_thresholds.numpy()
            pr_optimal_idx = np.nanargmax(pr_fscore)
            optimal_threshold = pr_thresholds_np[pr_optimal_idx]

            # Classify anomalies based on the optimal threshold
            anomalies = self.calculate_anomaly_scores(errors, optimal_threshold)

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels.int().numpy(), anomalies.numpy()).ravel()

            # Save plots if save_path is provided
            # if save_path is not None:
                # ts_save_dir = os.path.join(save_path, f"time_series_{ts_id}")
                # os.makedirs(ts_save_dir, exist_ok=True)

                # Plot ROC Curve
                # roc_save_path = os.path.join(ts_save_dir, f'roc_curve_ts_{ts_id}.pdf')
                # plot_roc_curve(
                #     fpr.numpy(), tpr.numpy(), roc_auc.item(),
                #     optimal_idx=torch.argmax(tpr - fpr).item(),  # Optimal index for plotting
                #     thresholds=roc_thresholds.numpy(),
                #     save_path=roc_save_path
                # )

                # Plot Precision-Recall curve
                # pr_save_path = os.path.join(ts_save_dir, f'pr_curve_ts_{ts_id}.pdf')
                # plot_precision_recall_curve(
                #     precision_vals.numpy(), recall_vals.numpy(),
                #     pr_auc.item(), pr_optimal_idx, pr_thresholds_np,
                #     save_path=pr_save_path
                # )

            return {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'pr_auc': pr_auc.item(),
                'roc_auc': roc_auc.item()  # Include ROC AUC
            }

        except Exception as e:
            Log.error(f"Error computing metrics for time series {ts_id}: {e}")
            return None

    @staticmethod
    def compute_reconstruction_errors(predictions, targets):
        """
        Computes the reconstruction errors for each sample in the dataset.

        Args:
            predictions (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_features].
            targets (torch.Tensor): True targets of shape [batch_size, seq_len, n_features].

        Returns:
            torch.Tensor: Reconstruction errors of shape [batch_size].
        """
        # Compute reconstruction errors per sample
        errors = torch.mean((predictions - targets) ** 2, dim=(1, 2))
        return errors.detach().cpu()

    @staticmethod
    def determine_thresholds(errors, labels):
        """
        Determines the optimal thresholds for anomaly detection using ROC and PR curves.

        Args:
            errors (torch.Tensor): Reconstruction errors for each sample.
            labels (torch.Tensor): True labels (1 for anomaly, 0 for normal).

        Returns:
            tuple: Contains ROC and PR curve information and optimal thresholds.
        """
        if torch.isnan(errors).any():
            Log.error("Errors contain NaN values. Cannot compute ROC or PR curves.")
            return None

        if len(labels.unique()) < 2:
            Log.error("Only one class present in labels. Cannot compute ROC or PR curves.")
            return None

        try:
            # Compute ROC curve
            fpr, tpr, roc_thresholds = roc(errors, labels.int(), task='binary')

            # Find optimal threshold for ROC
            roc_optimal_idx = torch.argmax(tpr - fpr)
            optimal_threshold = roc_thresholds[roc_optimal_idx]

            # Compute Precision-Recall curve
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(errors, labels.int(), task='binary')

            # Convert tensors to CPU
            fpr = fpr.cpu()
            tpr = tpr.cpu()
            roc_thresholds = roc_thresholds.cpu()
            precision_vals = precision_vals.cpu()
            recall_vals = recall_vals.cpu()
            pr_thresholds = pr_thresholds.cpu()

            return (
                optimal_threshold,
                fpr, tpr, roc_thresholds, None,
                precision_vals, recall_vals, pr_thresholds, None, None
            )

        except Exception as e:
            Log.error(f"Error computing ROC or PR curves: {e}")
            return None

    @staticmethod
    def calculate_anomaly_scores(errors, threshold):
        """
        Classifies samples as anomalies based on the threshold.

        Args:
            errors (torch.Tensor): Reconstruction errors for each sample.
            threshold (float): Threshold value for classifying anomalies.

        Returns:
            torch.Tensor: Binary tensor where 1 indicates an anomaly and 0 indicates normal.
        """
        anomalies = (errors >= threshold).int()
        return anomalies

    @staticmethod
    def calculate_evaluation_metrics(anomalies, labels):
        """
        Calculates evaluation metrics for anomaly detection.

        Args:
            anomalies (torch.Tensor): Predicted anomaly labels.
            labels (torch.Tensor): True anomaly labels.

        Returns:
            tuple: Precision, recall, and f1_score.
        """
        # Convert tensors to NumPy arrays
        anomalies_np = anomalies.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Use sklearn's precision_recall_fscore_support
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels_np, anomalies_np, average='binary')

        return precision, recall, f1_score
