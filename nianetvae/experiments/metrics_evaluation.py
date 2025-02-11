from typing import Any
import numpy as np
import torch
import torchmetrics
from torch import tensor, Tensor
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
from torchmetrics.utilities.checks import _check_same_shape

from log import Log


class EvaluationMetrics:
    def __init__(self):
        self.MAE_metric = torchmetrics.MeanAbsoluteError()  # Low is better
        self.MSE_metric = torchmetrics.MeanSquaredError()  # Low is better
        self.RMSE_metric = torchmetrics.MeanSquaredError(squared=False)  # Low is better
        self.MAPE_metric = torchmetrics.MeanAbsolutePercentageError() # Low is better
        self.RMAPE_metric = RootMeanAbsolutePercentageError() # Low is better

        # Initialize metrics with the worst possible values
        self.MAE = int(9e10)
        self.MSE = int(9e10)
        self.RMSE = int(9e10)
        self.MAPE = int(9e10)
        self.RMAPE = int(9e10)

    def to(self, device):
        self.MAE_metric.to(device)
        self.MSE_metric.to(device)
        self.RMSE_metric.to(device)
        self.MAPE_metric.to(device)
        self.RMAPE_metric.to(device)

    def update(self, predictions, targets):
        # Reshape predictions and targets
        reshaped_predictions = predictions.view(predictions.size(0), -1)
        reshaped_targets = targets.view(targets.size(0), -1)

        # Update metrics
        try:
            self.MAE_metric.update(reshaped_predictions, reshaped_targets)
            self.MSE_metric.update(reshaped_predictions, reshaped_targets)
            self.RMSE_metric.update(reshaped_predictions, reshaped_targets)
            self.MAPE_metric.update(reshaped_predictions, reshaped_targets)
            self.RMAPE_metric.update(reshaped_predictions, reshaped_targets)
        except Exception as e:
            Log.error(f"Error updating metrics: {e}")

    def compute(self):
        try:
            self.MAE = round(self.MAE_metric.compute().item(), 4)
            self.MSE = round(self.MSE_metric.compute().item(), 4)
            self.RMSE = round(self.RMSE_metric.compute().item(), 4)
            self.MAPE = round(self.MAPE_metric.compute().item(), 4)
            self.RMAPE = round(self.RMAPE_metric.compute().item(), 4)

        except Exception as e:
            Log.error(f"Error during raw metric computation: {e}")

        return {
            'MAE': self.MAE,
            'MSE': self.MSE,
            'RMSE': self.RMSE,
            'MAPE': self.MAPE,
            'RMAPE': self.RMAPE,
        }

    def are_metrics_complete(self):
        return all(
            metric_value is not None
            for metric_value in [self.MAE, self.MSE, self.RMSE, self.MAPE, self.RMAPE]
        )


class RootMeanAbsolutePercentageError(torchmetrics.Metric):
    """
    Compute Root Mean Absolute Percentage Error (RMAPE) by reusing MAPE.

    .. math:: \text{RMAPE} = \sqrt{\text{MAPE}}
    """

    def __init__(self):
        super().__init__()
        self.mape_metric = torchmetrics.MeanAbsolutePercentageError()

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the state by passing predictions and targets to the MAPE metric.

        Args:
            preds (Tensor): Predicted values.
            target (Tensor): Ground truth values.
        """
        self.mape_metric.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the Root Mean Absolute Percentage Error (RMAPE) by taking the square root of MAPE.

        Returns:
            Tensor: The RMAPE score.
        """
        return torch.sqrt(self.mape_metric.compute())