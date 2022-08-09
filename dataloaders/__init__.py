from dataloaders.time_series import ECG5000
from dataloaders.time_series import TimeSeriesDataset

__all__ = ["ECG5000", "TimeSeriesDataset"]
__import__("pkg_resources").declare_namespace(__name__)