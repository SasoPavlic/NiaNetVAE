import math

import torch
import torchmetrics
from torch import tensor, Tensor


class EvaluationMetrics:
    def __init__(self):
        self.ABS_REL_metric = AbsoluteRelativeDifference()  # Low is better
        self.CADL_metric = ConvAutoencoderDepthLoss()  # Low is better
        self.DELTA1_metric = Delta1()  # High is better
        self.DELTA2_metric = Delta2()  # High is better
        self.DELTA3_metric = Delta3()  # High is better
        self.LOG10_metric = Log10AbsoluteRelativeDifference()  # Low is better
        self.MAE_metric = torchmetrics.MeanAbsoluteError()  # Low is better
        self.MSE_metric = torchmetrics.MeanSquaredError()  # Low is better
        self.RMSE_metric = torchmetrics.MeanSquaredError()  # Low is better

        self.ABS_REL = None
        self.CADL = None
        self.DELTA1 = None
        self.DELTA2 = None
        self.DELTA3 = None
        self.LOG10 = None
        self.MAE = None
        self.MSE = None
        self.RMSE = None

    def to(self, device):
        self.ABS_REL_metric.to(device)
        self.CADL_metric.to(device)
        self.DELTA1_metric.to(device)
        self.DELTA2_metric.to(device)
        self.DELTA3_metric.to(device)
        self.LOG10_metric.to(device)
        self.MAE_metric.to(device)
        self.MSE_metric.to(device)
        self.RMSE_metric.to(device)

    def update(self, predictions, targets):
        self.ABS_REL_metric.update(predictions, targets)
        self.DELTA1_metric.update(predictions, targets)
        self.DELTA2_metric.update(predictions, targets)
        self.DELTA3_metric.update(predictions, targets)
        self.LOG10_metric.update(predictions, targets)
        self.MAE_metric.update(predictions, targets)
        self.MSE_metric.update(predictions, targets)
        self.RMSE_metric.update(predictions, targets)

    def update_CADL(self, batch_loss):
        self.CADL_metric.update(batch_loss)

    def compute(self):
        self.ABS_REL = self.ABS_REL_metric.compute().item()
        self.CADL = self.CADL_metric.compute().item()
        self.DELTA1 = self.DELTA1_metric.compute().item()
        self.DELTA2 = self.DELTA2_metric.compute().item()
        self.DELTA3 = self.DELTA3_metric.compute().item()
        self.LOG10 = self.LOG10_metric.compute().item()
        self.MAE = self.MAE_metric.compute().item()
        self.MSE = self.MSE_metric.compute().item()
        self.RMSE = torch.sqrt(self.RMSE_metric.compute()).item()

        return {
            'ABS_REL': self.ABS_REL,
            'CADL': self.CADL,
            'DELTA1': self.DELTA1,
            'DELTA2': self.DELTA2,
            'DELTA3': self.DELTA3,
            'LOG10': self.LOG10,
            'MAE': self.MAE,
            'MSE': self.MSE,
            'RMSE': self.RMSE
        }

    def are_metrics_complete(self):
        return all(metric_value is not None for metric_value in
                   [self.ABS_REL, self.CADL, self.DELTA1, self.DELTA2, self.DELTA3, self.LOG10, self.MAE, self.MSE,
                    self.RMSE])

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


class AbsoluteRelativeDifference(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("absolute_difference", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        absolute_difference = torch.abs(predictions - targets)
        denominator = torch.abs(predictions)

        # # Handle zero values
        # zero_mask = (tensor1 == 0) & (tensor2 == 0)
        # absolute_difference[zero_mask] = 0.0
        # denominator[zero_mask] = 1.0  # Avoid division by zero
        #
        # # Handle negative values
        # negative_mask = (tensor1 < 0) | (tensor2 < 0)
        # absolute_difference[negative_mask] = 0.0
        # denominator[negative_mask] = 1.0  # Avoid division by zero

        self.absolute_difference += torch.sum(absolute_difference)
        self.denominator += torch.sum(denominator)

    def compute(self):
        relative_difference = torch.zeros_like(self.denominator)
        non_zero_mask = self.denominator != 0
        relative_difference[non_zero_mask] = self.absolute_difference[non_zero_mask] / self.denominator[non_zero_mask]
        return relative_difference.mean()


class Log10AbsoluteRelativeDifference(torchmetrics.Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_log10_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        eps = torch.finfo(torch.float32).eps

        log10_predictions = torch.abs(predictions)
        log10_predictions = torch.log10(log10_predictions + eps)
        log10_targets = torch.log10(targets + eps)

        # # Handle zero values
        # zero_mask = (predictions == 0) & (targets == 0)
        # nonzero_mask = ~zero_mask
        # log10_predictions[zero_mask] = float('-inf')
        # log10_targets[zero_mask] = float('-inf')
        #
        # # Handle negative values
        # negative_mask = (predictions < 0) | (targets < 0)
        # positive_mask = ~negative_mask
        # log10_predictions[negative_mask] = float('-inf')
        # log10_targets[negative_mask] = float('-inf')
        #
        # # Handle infinity values
        # infinity_mask = (~torch.isfinite(log10_predictions)) | (~torch.isfinite(log10_targets))
        # log10_predictions[infinity_mask] = float('-inf')
        # log10_targets[infinity_mask] = float('-inf')

        absolute_difference = torch.abs(log10_predictions - log10_targets)
        relative_difference = absolute_difference / (torch.abs(log10_targets) + 1e-8)
        relative_difference[torch.isnan(relative_difference)] = 0.0

        self.sum_log10_diff += torch.sum(relative_difference)
        self.num_examples += predictions.numel()

    def compute(self):
        if self.num_examples == 0:
            return float('nan')

        return self.sum_log10_diff / self.num_examples


class ConvAutoencoderDepthLoss(torchmetrics.Metric):
    # https: // www.pytorchlightning.ai / blog / torchmetrics - pytorch - metrics - built - to - scale
    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_loss: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            batch_loss: Predictions from model for a given batch
        """

        self.sum_error += torch.sum(batch_loss)

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_error


class Delta1(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= self.threshold)
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count


class Delta2(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= math.pow(self.threshold, 2))
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count


class Delta3(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= math.pow(self.threshold, 3))
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count
