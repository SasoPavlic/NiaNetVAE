# anomaly_detection.py
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

from nianetvae.experiments.visualization import plot_roc_curve


def compute_reconstruction_errors(predictions, targets):
    """
    Computes the reconstruction errors for each sample in the dataset.

    Args:
        predictions (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_features].
        targets (torch.Tensor): True targets of shape [batch_size, seq_len, n_features].

    Returns:
        numpy.ndarray: Reconstruction errors of shape [num_samples].
    """
    # Convert tensors to numpy arrays
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Compute reconstruction errors per sample
    errors = np.mean((predictions - targets) ** 2, axis=(1, 2))
    return errors

def determine_threshold(errors, labels):
    """
    Determines the optimal threshold for anomaly detection using the ROC curve.

    Args:
        errors (numpy.ndarray): Reconstruction errors for each sample.
        labels (numpy.ndarray): True labels (1 for anomaly, 0 for normal).

    Returns:
        tuple: (optimal_threshold, fpr, tpr, thresholds, roc_auc, optimal_idx)
    """
    if np.isnan(errors).any():
        print("Errors contain NaN values. Cannot compute ROC curve.")
        return None, None, None, None, None, None

    if len(np.unique(labels)) < 2:
        print("Only one class present in labels. Cannot compute ROC curve.")
        return None, None, None, None, None, None

    try:
        fpr, tpr, thresholds = roc_curve(labels, errors)
        roc_auc = roc_auc_score(labels, errors)
        # Find the threshold that gives the best balance between TPR and FPR
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, fpr, tpr, thresholds, roc_auc, optimal_idx
    except ValueError as e:
        print(f"Error computing ROC curve: {e}")
        return None, None, None, None, None, None

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

def calculate_evaluation_metrics(anomalies, labels, roc_auc):
    """
    Calculates evaluation metrics for anomaly detection.

    Args:
        anomalies (numpy.ndarray): Predicted anomaly labels.
        labels (numpy.ndarray): True anomaly labels.
        roc_auc (float): ROC AUC score computed from continuous errors.

    Returns:
        dict: Dictionary containing precision, recall, f1_score, and roc_auc.
    """
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, anomalies, average='binary')
    return {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1_score, 3),
        'roc_auc': round(roc_auc, 3) if roc_auc is not None else None
    }

def perform_anomaly_detection(all_predictions, all_targets, all_labels, save_path=None):
    """
    Performs the entire anomaly detection process, including error computation,
    threshold determination, and metric calculation.

    Args:
        all_predictions (torch.Tensor): Concatenated model predictions.
        all_targets (torch.Tensor): Concatenated true targets.
        all_labels (torch.Tensor): Concatenated true labels.
        save_path (str): Directory to save plots, if any.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    errors = compute_reconstruction_errors(all_predictions, all_targets)
    labels = all_labels.detach().cpu().numpy()

    # Check for NaN values in errors
    if np.isnan(errors).any():
        print("Errors contain NaN values. Skipping anomaly detection for this model.")
        return {
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None
        }

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Only one class present in labels. Cannot compute ROC AUC.")
        return {
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None
        }

    threshold, fpr, tpr, thresholds, roc_auc, optimal_idx = determine_threshold(errors, labels)
    if threshold is None:
        print("Could not determine threshold due to errors in ROC computation.")
        return {
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None
        }

    anomalies = calculate_anomaly_scores(errors, threshold)
    metrics = calculate_evaluation_metrics(anomalies, labels, roc_auc=roc_auc)

    # Plot ROC curve if logger_dir is provided
    if save_path is not None:
        save_path = os.path.join(save_path, f'roc_curve{metrics["roc_auc"]}.pdf')
        plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, thresholds, save_path=save_path)

    return metrics
