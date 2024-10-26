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
    plt.figure()
    lw = 2
    x = fpr[optimal_idx]
    y = tpr[optimal_idx]

    point1 = [0, 1]
    point2 = [x, y]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    distance = round(np.sqrt((x - point1[0])**2 + (y - point1[1])**2), 2)
    print(f"Distance: {distance}")

    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="Recurrent VAE (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--", label='Random classifier (AUC = 0.50)')
    plt.plot(x_values, y_values, color="red", lw=lw, linestyle=":", label=f'Distance = {distance}')
    plt.plot(x, y, '-ro', label=f'Optimal threshold (FPR={round(x,2)}, TPR={round(y,2)})')

    # Annotate the optimal point
    plt.annotate('(%.2f, %.2f)' % (x, y), xy=(x, y), ha='center')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.show()
