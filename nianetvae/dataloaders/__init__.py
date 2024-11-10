from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from log import Log


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")


class BaseDataLoader(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            data_path: str,
            batch_size: int,
            seq_len: int,
            num_workers: int,
            persistent_workers: bool,
            pin_memory: bool,
            train_size: float,
            val_size: float,
            test_size: float,
            data_percentage: float,
            **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.data_percentage = data_percentage

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def _empty_dataloader(self):
        return DataLoader(
            EmptyDataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def train_dataloader(self):
        if self.train_dataset:
            # Return a DataLoader for the training dataset
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
        else:
            Log.warning("Train dataset is None. Returning an empty DataLoader.")
            return self._empty_dataloader()

    def val_dataloader(self):
        if self.val_dataset:
            # Return a DataLoader for the validation dataset
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
        else:
            Log.warning("Validation dataset is None. Returning an empty DataLoader.")
            return self._empty_dataloader()

    def test_dataloader(self):
        if self.test_dataset:
            # Return a DataLoader for the test dataset
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
        else:
            Log.warning("Test dataset is None. Returning an empty DataLoader.")
            return self._empty_dataloader()
