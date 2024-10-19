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
        self.R2_metric = torchmetrics.R2Score(
            num_outputs=num_outputs, multioutput='uniform_average'
        )  # High is better

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
        # Reshape predictions and targets for metrics that require specific dimensions
        reshaped_predictions = predictions.view(predictions.size(0), -1)
        reshaped_targets = targets.view(targets.size(0), -1)

        # Update MAE
        self.MAE_metric.update(reshaped_predictions, reshaped_targets)
        print("Updated MAE_metric")

        # Update MSE
        self.MSE_metric.update(reshaped_predictions, reshaped_targets)
        print("Updated MSE_metric")

        # Update RMSE
        self.RMSE_metric.update(reshaped_predictions, reshaped_targets)
        print("Updated RMSE_metric")

        # Update R2 Score
        self.R2_metric.update(reshaped_predictions, reshaped_targets)
        print("Updated R2_metric")

        # Check if data is univariate
        if predictions.size(-1) == 1:
            self.DTW_metric.update(predictions, targets)
            print("Updated DTW_metric")
        else:
            print("Skipping DTW_metric for multivariate data")

    def compute(self):
        self.MAE = self.MAE_metric.compute().item()
        self.MSE = self.MSE_metric.compute().item()
        self.RMSE = self.RMSE_metric.compute().item()
        self.R2 = self.R2_metric.compute().item()

        # Check if DTW metric has been updated
        dtw_value = self.DTW_metric.compute()
        if torch.isnan(dtw_value):
            self.DTW = float(0.0)
        else:
            self.DTW = dtw_value.item()

        return {
            'MAE': self.MAE,
            'MSE': self.MSE,
            'RMSE': self.RMSE,
            'DTW': self.DTW,
            'R2': self.R2,
        }

    def are_metrics_complete(self):
        return all(
            metric_value is not None
            for metric_value in [self.MAE, self.MSE, self.RMSE, self.DTW, self.R2]
        )

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
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        batch_size = predictions.shape[0]
        for i in range(batch_size):
            distance = self._dtw(predictions[i], targets[i])
            self.dtw_distance += torch.tensor(distance)
            self.num_samples += 1
        print(f"Updated DTW_metric for batch of size {batch_size}")

    def compute(self):
        if self.num_samples > 0:
            return self.dtw_distance / self.num_samples
        else:
            return torch.tensor(float('nan'))

    def _dtw(self, x, y):
        # x and y have shape [seq_len, n_features]
        x = x.flatten()
        y = y.flatten()

        # Initialize the cost matrix
        n, m = len(x), len(y)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        # Populate the cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = (x[i - 1] - y[j - 1]) ** 2
                cost[i, j] = dist + min(
                    cost[i - 1, j],    # Insertion
                    cost[i, j - 1],    # Deletion
                    cost[i - 1, j - 1]  # Match
                )

        return cost[n, m]
