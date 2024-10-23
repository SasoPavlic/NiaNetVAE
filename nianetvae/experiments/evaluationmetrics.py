import numpy as np
import torch
import torchmetrics
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError
from log import Log  # Ensure you have your custom Log module imported


class EvaluationMetrics:
    def __init__(self, num_outputs):
        self.MAE_metric = torchmetrics.MeanAbsoluteError()  # Low is better
        self.MSE_metric = torchmetrics.MeanSquaredError()  # Low is better
        self.RMSE_metric = torchmetrics.MeanSquaredError(squared=False)  # Low is better
        self.DTW_metric = DynamicTimeWarping()  # Low is better
        self.R2_metric = torchmetrics.R2Score(num_outputs=num_outputs, multioutput='uniform_average')  # High is better

        # Initialize metrics with worst possible values
        self.MAE = int(9e10)
        self.MSE = int(9e10)
        self.RMSE = int(9e10)
        self.DTW = int(9e10)
        self.R2 = float('-inf')

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
        try:
            self.MAE_metric.update(reshaped_predictions, reshaped_targets)
        except Exception as e:
            Log.error(f"Error updating MAE_metric: {e}")
            self.MAE_metric = None  # Mark as None to skip computation

        # Update MSE
        try:
            self.MSE_metric.update(reshaped_predictions, reshaped_targets)
        except Exception as e:
            Log.error(f"Error updating MSE_metric: {e}")
            self.MSE_metric = None

        # Update RMSE
        try:
            self.RMSE_metric.update(reshaped_predictions, reshaped_targets)
        except Exception as e:
            Log.error(f"Error updating RMSE_metric: {e}")
            self.RMSE_metric = None

        # Update R2 Score
        try:
            self.R2_metric.update(reshaped_predictions, reshaped_targets)
        except Exception as e:
            Log.error(f"Error updating R2_metric: {e}")
            self.R2_metric = None

        # Check if data is univariate
        if predictions.size(-1) == 1:
            try:
                self.DTW_metric.update(predictions, targets)
            except Exception as e:
                Log.error(f"Error updating DTW_metric: {e}")
                self.DTW_metric = None
        else:
            print("Skipping DTW_metric for multivariate data")

    def compute(self):
        # Compute MAE
        try:
            if self.MAE_metric is not None:
                self.MAE = self.MAE_metric.compute().item()
        except Exception as e:
            Log.error(f"Error computing MAE_metric: {e}")
            self.MAE = int(9e10)

        # Compute MSE
        try:
            if self.MSE_metric is not None:
                self.MSE = self.MSE_metric.compute().item()
        except Exception as e:
            Log.error(f"Error computing MSE_metric: {e}")
            self.MSE = int(9e10)

        # Compute RMSE
        try:
            if self.RMSE_metric is not None:
                self.RMSE = self.RMSE_metric.compute().item()
        except Exception as e:
            Log.error(f"Error computing RMSE_metric: {e}")
            self.RMSE = int(9e10)

        # Compute R2
        try:
            if self.R2_metric is not None:
                self.R2 = self.R2_metric.compute().item()
        except Exception as e:
            Log.error(f"Error computing R2_metric: {e}")
            self.R2 = 0.0

        # Compute DTW only for Univariate dataset
        try:
            if self.DTW_metric is not None:
                dtw_value = self.DTW_metric.compute()
                if torch.isnan(dtw_value):
                    self.DTW = int(9e10)
                else:
                    self.DTW = dtw_value.item()
        except Exception as e:
            Log.error(f"Error computing DTW_metric: {e}")
            self.DTW = int(9e10)

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
        if range_value == 0:
            Log.error("Normalization range is zero.")
            return 0.0
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
            try:
                distance = self._dtw(predictions[i], targets[i])
                self.dtw_distance += torch.tensor(distance)
                self.num_samples += 1
            except Exception as e:
                Log.error(f"Error computing DTW for sample {i}: {e}")

    def compute(self):
        if self.num_samples > 0:
            return self.dtw_distance / self.num_samples
        else:
            return torch.tensor(int(9e10))  # Worst possible value for DTW

    def _dtw(self, x, y):
        # x and y have shape [seq_len, n_features]
        x = x.flatten()
        y = y.flatten()

        n, m = len(x), len(y)

        # Early exit if sequences are empty
        if n == 0 or m == 0:
            Log.error("One of the sequences is empty in DTW computation.")
            return int(9e10)  # Worst possible value for DTW

        # Initialize the cost matrix
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0

        # Populate the cost matrix
        try:
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    dist = (x[i - 1] - y[j - 1]) ** 2
                    cost[i, j] = dist + min(
                        cost[i - 1, j],    # Insertion
                        cost[i, j - 1],    # Deletion
                        cost[i - 1, j - 1]  # Match
                    )
            return cost[n, m]
        except Exception as e:
            Log.error(f"Error in DTW computation: {e}")
            return int(9e10)  # Worst possible value for DTW
