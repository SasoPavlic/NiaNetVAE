import os
from typing import Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from scipy.io import arff
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'signal': self.data[index],
            'target': self.targets[index]
        }
        return sample


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


class ECG5000DataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(self.data_path, 'ECG5000_TRAIN.arff')) as f:
            train_data, train_meta = arff.loadarff(f)
        with open(os.path.join(self.data_path, 'ECG5000_TEST.arff')) as f:
            test_data, test_meta = arff.loadarff(f)

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Combine the train and test datasets
        combined_df = pd.concat([train_df, test_df])

        # Apply data percentage filter
        combined_df = combined_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        combined_data = combined_df.drop(columns=['target']).values
        combined_target = pd.to_numeric(combined_df['target']).values

        # Split the combined data into train, validation, and test sets
        x_train, x_test, y_train, y_test = train_test_split(combined_data, combined_target, test_size=self.test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size)

        self.train_dataset = BaseDataset(x_train, y_train)
        self.val_dataset = BaseDataset(x_val, y_val)
        self.test_dataset = BaseDataset(x_test, y_test)
