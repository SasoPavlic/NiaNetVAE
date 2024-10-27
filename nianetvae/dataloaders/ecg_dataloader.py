# ecg_dataloader.py

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class ECG5000Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        # Reshape data if necessary
        if signal.dim() == 1:
            # Univariate data: add an extra dimension
            signal = signal.unsqueeze(-1)

        target = self.targets[idx]
        return {'signal': signal, 'target': target.int()}


class ECG5000DataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        # Load training data
        with open(os.path.join(self.data_path, 'ECG5000_TRAIN.arff')) as f:
            train_data, train_meta = arff.loadarff(f)
        # Load testing data
        with open(os.path.join(self.data_path, 'ECG5000_TEST.arff')) as f:
            test_data, test_meta = arff.loadarff(f)

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Combine the train and test datasets
        combined_df = pd.concat([train_df, test_df])

        # Apply data percentage filter
        combined_df = combined_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        # Separate features and targets
        combined_data = combined_df.drop(columns=['target']).values
        combined_target = pd.to_numeric(combined_df['target']).values

        # Map multiclass labels to binary labels: 0 for normal, 1 for anomaly
        # In the ECG5000 dataset, class '1' is normal, and the rest are anomalies
        combined_target = np.where(combined_target == 1, 0, 1)

        # Calculate sizes for train, validation, and test sets
        total_size = len(combined_data)
        test_size = int(total_size * self.test_size / 100)
        val_size = int(total_size * self.val_size / 100)
        train_size = total_size - test_size - val_size

        # Split the combined data into train, validation, and test sets
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            combined_data, combined_target, test_size=test_size, random_state=42, stratify=combined_target
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
        )

        self.train_dataset = ECG5000Dataset(x_train, y_train)
        self.val_dataset = ECG5000Dataset(x_val, y_val)
        self.test_dataset = ECG5000Dataset(x_test, y_test)

        Log.info(f"Train size: {len(self.train_dataset)}")
        Log.info(f"Validation size: {len(self.val_dataset)}")
        Log.info(f"Test size: {len(self.test_dataset)}")

        # Log the distribution of classes in each set
        train_counts = pd.Series(y_train).value_counts()
        val_counts = pd.Series(y_val).value_counts()
        test_counts = pd.Series(y_test).value_counts()
        Log.info(f"Train class distribution: {train_counts.to_dict()}")
        Log.info(f"Validation class distribution: {val_counts.to_dict()}")
        Log.info(f"Test class distribution: {test_counts.to_dict()}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )
