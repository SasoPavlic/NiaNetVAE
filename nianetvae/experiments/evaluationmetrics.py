import numpy as np
import torch
import torchmetrics
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError


class EvaluationMetrics:
    def __init__(self, num_outputs):
        self.MAE_metric = torchmetrics.MeanAbsoluteError()  # Low is better
        self.MSE_metric = torchmetrics.MeanSquaredError()  # Low is better
        self.RMSE_metric = torchmetrics.MeanSquaredError(squared=False)  # Low is better
        self.DTW_metric = DynamicTimeWarping()  # Low is better
        self.R2_metric = torchmetrics.R2Score(num_outputs=num_outputs, multioutput='uniform_average')  # High is better

        self.MAE = None
        self.MSE = None
        self.RMSE = None
        self.DTW = None
        self.R2 = None

    def to(self, device):
        self.MAE_metric.to(device)
        self.MSE_metric.to(device)
        self.RMSE_metric.to(device)
        self.DTW_metric.to(device)
        self.R2_metric.to(device)

    def update(self, predictions, targets):
        self.MAE_metric.update(predictions, targets)
        self.MSE_metric.update(predictions, targets)
        self.RMSE_metric.update(predictions, targets)
        self.DTW_metric.update(predictions, targets)
        self.R2_metric.update(predictions, targets)

    def compute(self):
        self.MAE = self.MAE_metric.compute().item()
        self.MSE = self.MSE_metric.compute().item()
        self.RMSE = torch.sqrt(self.RMSE_metric.compute()).item()
        self.DTW = self.DTW_metric.compute().item()
        self.R2 = self.R2_metric.compute().item()

        return {
            'MAE': self.MAE,
            'MSE': self.MSE,
            'RMSE': self.RMSE,
            'DTW': self.DTW,
            'R2': self.R2,
        }

    def are_metrics_complete(self):
        return all(metric_value is not None for metric_value in
                   [self.MAE, self.MSE, self.RMSE, self.DTW, self.R2])


    def normalize(self, value, min_value, max_value):
        """
        Normalize a value to a range between 0 and 1 using min-max normalization.

        Args:
            value (float): The value to be normalized.
            min_value (float): The minimum value observed for the variable.
            max_value (float): The maximum value observed for the variable.

        Returns:
            float: The normalized value between 0 and 1.
        """
        range_value = max_value - min_value
        normalized_value = (value - min_value) / range_value
        return normalized_value


class DynamicTimeWarping(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("dtw_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        # Ensure predictions and targets are on the CPU and converted to numpy arrays
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        batch_size = predictions.shape[0]
        for i in range(batch_size):
            self.dtw_distance += self._dtw(predictions[i], targets[i])
            self.num_samples += 1

    def compute(self):
        return self.dtw_distance / self.num_samples

    def _dtw(self, x, y):
        # Initialize the cost matrix
        n, m = len(x), len(y)
        cost = np.full((n, m), np.inf)
        cost[0, 0] = 0

        # Populate the cost matrix
        for i in range(n):
            for j in range(m):
                dist = (x[i] - y[j]) ** 2
                if i > 0:
                    cost[i, j] = min(cost[i, j], cost[i - 1, j] + dist)
                if j > 0:
                    cost[i, j] = min(cost[i, j], cost[i, j - 1] + dist)
                if i > 0 and j > 0:
                    cost[i, j] = min(cost[i, j], cost[i - 1, j - 1] + dist)

        return cost[n - 1, m - 1]
