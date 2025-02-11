# visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(fpr, tpr, roc_auc, optimal_idx, thresholds, save_path='roc_curve.pdf'):
    """
    Plots the ROC curve with the optimal threshold point and other annotations.

    Args:
        fpr (array-like): False positive rates.
        tpr (array-like): True positive rates.
        roc_auc (float): Area under the ROC curve.
        optimal_idx (int): Index of the optimal threshold in fpr, tpr arrays.
        thresholds (array-like): Thresholds used to compute fpr and tpr.
        save_path (str): Path to save the plot.
    """
    # Ensure variables are numpy arrays
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    thresholds = np.array(thresholds)

    # Round variables to 4 decimals
    fpr = np.round(fpr, 4)
    tpr = np.round(tpr, 4)
    thresholds = np.round(thresholds, 4)
    roc_auc = round(roc_auc, 4)

    plt.figure()
    lw = 2
    x = fpr[optimal_idx]
    y = tpr[optimal_idx]

    # Round x and y to 4 decimals
    x = round(x, 4)
    y = round(y, 4)

    point1 = [0, 1]  # Ideal point in ROC space
    point2 = [x, y]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    distance = round(np.sqrt((x - point1[0]) ** 2 + (y - point1[1]) ** 2), 4)
    #print(f"Distance: {distance:.3f}")

    plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"Recurrent VAE (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label='Random Classifier (AUC = 0.500)')
    plt.plot(x_values, y_values, color="red", lw=lw, linestyle=":", label=f'Distance = {distance:.3f}')
    plt.plot(x, y, '-ro', label=f'Optimal Threshold (FPR={x:.3f}, TPR={y:.3f})')

    # Annotate the optimal point
    plt.annotate(f'({x:.3f}, {y:.3f})', xy=(x, y), ha='center')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    #plt.show()
    plt.close()


def plot_precision_recall_curve(precision, recall, pr_auc, optimal_idx, thresholds, save_path='pr_curve.pdf'):
    """
    Plots the Precision-Recall curve with the optimal threshold point and other annotations.

    Args:
        precision (array-like): Precision values.
        recall (array-like): Recall values.
        pr_auc (float): Area under the Precision-Recall curve.
        optimal_idx (int): Index of the optimal threshold in precision, recall arrays.
        thresholds (array-like): Thresholds used to compute precision and recall.
        save_path (str): Path to save the plot.
    """
    # Ensure variables are numpy arrays
    precision = np.array(precision)
    recall = np.array(recall)
    thresholds = np.array(thresholds)

    # Round variables to 4 decimals
    precision = np.round(precision, 4)
    recall = np.round(recall, 4)
    thresholds = np.round(thresholds, 4)
    pr_auc = round(pr_auc, 4)

    plt.figure()
    lw = 2
    x = recall[optimal_idx]
    y = precision[optimal_idx]

    # Round x and y to 4 decimals
    x = round(x, 4)
    y = round(y, 4)

    point1 = [1, 1]  # Ideal point in PR space
    point2 = [x, y]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    distance = round(np.sqrt((x - point1[0]) ** 2 + (y - point1[1]) ** 2), 4)
    #print(f"Distance: {distance:.3f}")

    plt.plot(recall, precision, color="darkorange", lw=lw, label=f"Recurrent VAE (AUC = {pr_auc:.4f})")
    plt.plot(x_values, y_values, color="red", lw=lw, linestyle=":", label=f'Distance = {distance:.4f}')
    plt.plot(x, y, '-ro', label=f'Optimal Threshold (Recall={x:.3f}, Precision={y:.4f})')

    # Annotate the optimal point
    plt.annotate(f'({x:.4f}, {y:.4f})', xy=(x, y), ha='center')

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    #plt.show()
    plt.close()
