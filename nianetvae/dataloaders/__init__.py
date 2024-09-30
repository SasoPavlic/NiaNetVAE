# from nianetvae.dataloaders.ecg_dataloader import ECG5000_train
# from nianetvae.dataloaders.ecg_dataloader import ECG5000_val
# from nianetvae.dataloaders.ecg_dataloader import ECG5000_test
# from nianetvae.dataloaders.ecg_dataloader import TimeSeriesDataset
#
# __all__ = ["ECG5000_train", "ECG5000_val", "ECG5000_test", "TimeSeriesDataset"]
# __import__("pkg_resources").declare_namespace(__name__)
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional

class BaseDataLoader(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            train_size: float,
            val_size: float,
            test_size: float,
            data_percentage: float,
            **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.data_percentage = data_percentage

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)