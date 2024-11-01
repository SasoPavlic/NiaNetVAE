# anomaly_detection.py

import os
import torch
import numpy as np
from torchmetrics.functional import (
    auroc,
    roc,
    precision_recall_curve,
    average_precision,
)
from sklearn.metrics import precision_recall_fscore_support  # Reintroduced sklearn import
from log import Log
from nianetvae.experiments.visualization import plot_roc_curve, plot_precision_recall_curve


class AnomalyDetectionMetrics:
    def __init__(self):
        # Initialize lists to store errors and labels
        self.errors = []
        self.labels = []

    def to(self, device):
        # If you plan to store tensors, move them to the specified device
        pass  # In this case, we're storing data on CPU

    def update(self, predictions, targets, labels):
        """
        Update internal state with new batch data.

        Args:
            predictions (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_features].
            targets (torch.Tensor): True targets of shape [batch_size, seq_len, n_features].
            labels (torch.Tensor): True labels of shape [batch_size].
        """
        # Compute reconstruction errors for the batch
        batch_errors = self.compute_reconstruction_errors(predictions, targets)
        # Ensure labels are on CPU
        batch_labels = labels.detach().cpu()
        # Append to lists
        self.errors.append(batch_errors)
        self.labels.append(batch_labels)

    def compute(self, save_path=None):
        """
        Compute the anomaly detection metrics based on accumulated data.

        Args:
            save_path (str): Directory to save plots, if any.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Concatenate all errors and labels
        errors = torch.cat(self.errors)
        labels = torch.cat(self.labels)

        # Check for NaN values in errors
        if torch.isnan(errors).any():
            Log.debug("Errors contain NaN values. Skipping anomaly detection for this model.")
            return {
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'pr_auc': None
            }

        if len(labels.unique()) < 2:
            Log.debug("Only one class present in labels. Cannot compute ROC AUC or PR AUC.")
            return {
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'pr_auc': None
            }

        # Determine thresholds and metrics
        threshold_info = self.determine_thresholds(errors, labels)
        if threshold_info is None:
            Log.debug("Could not determine thresholds due to errors in ROC or PR curve computation.")
            return {
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'pr_auc': None
            }

        (
            optimal_threshold,
            fpr, tpr, roc_thresholds, roc_auc, roc_optimal_idx,
            precision_vals, recall_vals, pr_thresholds, pr_auc, pr_optimal_idx
        ) = threshold_info

        # Classify anomalies based on the optimal threshold from ROC curve
        anomalies = self.calculate_anomaly_scores(errors, optimal_threshold)
        metrics = self.calculate_evaluation_metrics(anomalies, labels, roc_auc=roc_auc, pr_auc=pr_auc)

        # Plot ROC and PR curves if save_path is provided
        if save_path is not None:
            # Plot ROC curve
            roc_save_path = os.path.join(save_path, f'roc_curve_{metrics["roc_auc"]}.pdf')
            plot_roc_curve(fpr, tpr, roc_auc, roc_optimal_idx, roc_thresholds, save_path=roc_save_path)

            # Plot Precision-Recall curve
            pr_save_path = os.path.join(save_path, f'pr_curve_{metrics["pr_auc"]}.pdf')
            plot_precision_recall_curve(precision_vals, recall_vals, pr_auc, pr_optimal_idx, pr_thresholds, save_path=pr_save_path)

        return metrics

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
            # Compute ROC curve and AUC
            fpr, tpr, roc_thresholds = roc(errors, labels.int(), task='binary')
            roc_auc = auroc(errors, labels.int(), task='binary')

            # Convert tensors to numpy arrays for plotting and processing
            fpr = fpr.cpu().numpy()
            tpr = tpr.cpu().numpy()
            roc_thresholds = roc_thresholds.cpu().numpy()

            # Find optimal threshold for ROC
            roc_optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = roc_thresholds[roc_optimal_idx]

            # Compute Precision-Recall curve and AUC
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(errors, labels.int(), task='binary')
            pr_auc = average_precision(errors, labels.int(), task='binary')

            # Convert tensors to numpy arrays
            precision_vals = precision_vals.cpu().numpy()
            recall_vals = recall_vals.cpu().numpy()
            pr_thresholds = pr_thresholds.cpu().numpy()

            # Compute F1 scores to find optimal threshold
            pr_fscore = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-10)
            pr_optimal_idx = np.nanargmax(pr_fscore)

            # Round variables to 3 decimals
            optimal_threshold = round(float(optimal_threshold), 3)
            roc_auc = round(float(roc_auc), 3)
            pr_auc = round(float(pr_auc), 3)

            # Round arrays for plotting (optional)
            fpr = np.round(fpr, 3)
            tpr = np.round(tpr, 3)
            roc_thresholds = np.round(roc_thresholds, 3)
            precision_vals = np.round(precision_vals, 3)
            recall_vals = np.round(recall_vals, 3)
            pr_thresholds = np.round(pr_thresholds, 3)

            return (
                optimal_threshold,
                fpr, tpr, roc_thresholds, roc_auc, roc_optimal_idx,
                precision_vals, recall_vals, pr_thresholds, pr_auc, pr_optimal_idx
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
    def calculate_evaluation_metrics(anomalies, labels, roc_auc, pr_auc):
        """
        Calculates evaluation metrics for anomaly detection.

        Args:
            anomalies (torch.Tensor): Predicted anomaly labels.
            labels (torch.Tensor): True anomaly labels.
            roc_auc (float): ROC AUC score computed from continuous errors.
            pr_auc (float): PR AUC score computed from continuous errors.

        Returns:
            dict: Dictionary containing precision, recall, f1_score, roc_auc, and pr_auc.
        """
        # Convert tensors to NumPy arrays
        anomalies_np = anomalies.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Use sklearn's precision_recall_fscore_support
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels_np, anomalies_np, average='binary')

        # Round metrics to 3 decimals
        precision = round(precision, 3)
        recall = round(recall, 3)
        f1_score = round(f1_score, 3)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
