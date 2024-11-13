import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from log import Log
from nianetvae.dataloaders import BaseDataLoader


class ECG5000Dataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        if signal.dim() == 1:
            signal = signal.unsqueeze(-1)  # Ensure shape is [seq_len, 1]
        target = self.targets[idx]
        return {'signal': signal, 'target': target.int()}


class ECG5000DataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        train_data, train_labels, test_data, test_labels = self._load_data_files()

        train_data, train_labels = self._normalize_and_filter(train_data, train_labels, fit=True)
        test_data, test_labels = self._normalize_and_filter(test_data, test_labels, fit=False)

        self._split_train_validation(train_data, train_labels)

        self.test_dataset = ECG5000Dataset(test_data, test_labels)
        Log.info(f"Total test sequences: {len(self.test_dataset)}")
        Log.info(f"Test class distribution: {pd.Series(test_labels).value_counts().to_dict()}")

    def _load_data_files(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with open(os.path.join(self.data_path, 'ECG5000_TRAIN.arff')) as f:
            train_data, _ = arff.loadarff(f)
        with open(os.path.join(self.data_path, 'ECG5000_TEST.arff')) as f:
            test_data, _ = arff.loadarff(f)

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        train_data = train_df.drop(columns=['target']).values
        train_labels = pd.to_numeric(train_df['target']).values
        test_data = test_df.drop(columns=['target']).values
        test_labels = pd.to_numeric(test_df['target']).values

        train_labels = np.where(train_labels == 1, 0, 1)
        test_labels = np.where(test_labels == 1, 0, 1)

        return train_data, train_labels, test_data, test_labels

    def _normalize_and_filter(self, data: np.ndarray, labels: np.ndarray, fit=True) -> Tuple[np.ndarray, np.ndarray]:
        if data.size == 0:
            return np.array([]), np.array([])

        if fit:
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        else:
            if not hasattr(self, 'scaler'):
                raise AttributeError("Scaler not initialized. Ensure training data is normalized before test data.")

        data = self.scaler.transform(data)

        if self.data_percentage < 100:
            num_samples = int(data.shape[0] * (self.data_percentage / 100))
            data = data[:num_samples]
            labels = labels[:num_samples]

        return data, labels

    def _split_train_validation(self, data: np.ndarray, labels: np.ndarray) -> None:
        if self.val_size > 0:
            val_size = self.val_size / 100.0
            x_train, x_val, y_train, y_val = train_test_split(
                data, labels, test_size=val_size, random_state=42, stratify=labels
            )
            self.train_dataset = ECG5000Dataset(x_train, y_train)
            self.val_dataset = ECG5000Dataset(x_val, y_val)
            Log.info(f"Total training sequences: {len(self.train_dataset)}")
            Log.info(f"Total validation sequences: {len(self.val_dataset)}")
            Log.info(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
            Log.info(f"Validation class distribution: {pd.Series(y_val).value_counts().to_dict()}")
        else:
            self.train_dataset = ECG5000Dataset(data, labels)
            self.val_dataset = None
            Log.warning("Validation dataset is not created as val_size is set to 0.")
            Log.info(f"Train class distribution: {pd.Series(labels).value_counts().to_dict()}")
