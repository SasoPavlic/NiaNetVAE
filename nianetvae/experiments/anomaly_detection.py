# anomaly_detection.py
import os
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from log import Log
from nianetvae.experiments.visualization import plot_roc_curve, plot_precision_recall_curve


class AnomalyDetectionMetrics:
    def __init__(self):
        # Initialize lists to store errors and labels
        self.errors = []
        self.labels = []

    def to(self, device):
        # If you plan to store tensors, move them to the specified device
        pass  # In this case, we're storing data on CPU as numpy arrays

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
        # Convert labels to numpy
        batch_labels = labels.detach().cpu().numpy()
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
        errors = np.concatenate(self.errors)
        labels = np.concatenate(self.labels)

        # Check for NaN values in errors
        if np.isnan(errors).any():
            Log.debug("Errors contain NaN values. Skipping anomaly detection for this model.")
            return {
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'pr_auc': None
            }

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
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
            numpy.ndarray: Reconstruction errors of shape [batch_size].
        """
        # Convert tensors to numpy arrays
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # Compute reconstruction errors per sample
        errors = np.mean((predictions - targets) ** 2, axis=(1, 2))
        return errors

    @staticmethod
    def determine_thresholds(errors, labels):
        """
        Determines the optimal thresholds for anomaly detection using ROC and PR curves.

        Args:
            errors (numpy.ndarray): Reconstruction errors for each sample.
            labels (numpy.ndarray): True labels (1 for anomaly, 0 for normal).

        Returns:
            tuple: Contains ROC and PR curve information and optimal thresholds.
        """
        if np.isnan(errors).any():
            Log.error("Errors contain NaN values. Cannot compute ROC or PR curves.")
            return None

        if len(np.unique(labels)) < 2:
            Log.error("Only one class present in labels. Cannot compute ROC or PR curves.")
            return None

        try:
            # ROC Curve and AUC
            fpr, tpr, roc_thresholds = roc_curve(labels, errors)
            roc_auc = roc_auc_score(labels, errors)
            # Find the optimal threshold for ROC
            roc_optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = roc_thresholds[roc_optimal_idx]

            # Precision-Recall Curve and AUC
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(labels, errors)
            pr_auc = average_precision_score(labels, errors)
            # Find the optimal threshold for PR Curve
            pr_fscore = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-10)
            pr_optimal_idx = np.nanargmax(pr_fscore)
            # Note: We keep the optimal_threshold from ROC for consistency

            return (
                optimal_threshold,
                fpr, tpr, roc_thresholds, roc_auc, roc_optimal_idx,
                precision_vals, recall_vals, pr_thresholds, pr_auc, pr_optimal_idx
            )
        except ValueError as e:
            Log.error(f"Error computing ROC or PR curves: {e}")
            return None

    @staticmethod
    def calculate_anomaly_scores(errors, threshold):
        """
        Classifies samples as anomalies based on the threshold.

        Args:
            errors (numpy.ndarray): Reconstruction errors for each sample.
            threshold (float): Threshold value for classifying anomalies.

        Returns:
            numpy.ndarray: Binary array where 1 indicates an anomaly and 0 indicates normal.
        """
        anomalies = (errors >= threshold).astype(int)
        return anomalies

    @staticmethod
    def calculate_evaluation_metrics(anomalies, labels, roc_auc, pr_auc):
        """
        Calculates evaluation metrics for anomaly detection.

        Args:
            anomalies (numpy.ndarray): Predicted anomaly labels.
            labels (numpy.ndarray): True anomaly labels.
            roc_auc (float): ROC AUC score computed from continuous errors.
            pr_auc (float): PR AUC score computed from continuous errors.

        Returns:
            dict: Dictionary containing precision, recall, f1_score, roc_auc, and pr_auc.
        """
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels, anomalies, average='binary')
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1_score, 3),
            'roc_auc': round(roc_auc, 3) if roc_auc is not None else None,
            'pr_auc': round(pr_auc, 3) if pr_auc is not None else None
        }
