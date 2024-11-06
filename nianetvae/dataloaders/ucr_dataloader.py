import torch
from torch.utils.data import Dataset, DataLoader


class UCRDataset(Dataset):
    def __init__(self, data, targets, seq_len=200, stride=1):
        self.data = torch.tensor(data).float()  # Shape: [num_samples, num_features]
        self.targets = torch.tensor(targets).float()  # Shape: [num_samples]
        self.seq_len = seq_len
        self.stride = stride
        self.sequences, self.labels = self._create_sequences()

    def _create_sequences(self):
        if len(self.data) < self.seq_len:
            print(f"Data length ({len(self.data)}) is less than seq_len ({self.seq_len}). No sequences will be created.")
            return torch.empty((0, self.seq_len)), torch.empty((0,))

        sequences = []
        seq_labels = []
        num_samples = len(self.data)
        for i in range(0, num_samples - self.seq_len + 1, self.stride):
            sequence = self.data[i:i + self.seq_len]  # Shape: [seq_len]
            label = 1 if self.targets[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)

        if not sequences:
            print(f"No sequences were created for data length {len(self.data)} with seq_len {self.seq_len}.")
            return torch.empty((0, self.seq_len)), torch.empty((0,))

        return torch.stack(sequences), torch.tensor(seq_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.sequences[idx]  # Shape: [seq_len]
        target = self.labels[idx].int()
        return {'signal': signal.unsqueeze(-1), 'target': target}  # Add feature dimension

import os
import re
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from nianetvae.dataloaders import BaseDataLoader


class UCRDataLoader(BaseDataLoader):
    def __init__(self, dataset_type: str, data_path: str, seq_len: int, data_percentage: float = 100.0,
                 batch_size: int = 32, train_size=70, val_size: float = 10.0, test_size: float = 20.0,
                 num_workers: int = 0, pin_memory: bool = False, persistent_workers: bool = False,
                 filename: str = None, **kwargs):
        super().__init__(dataset_type=dataset_type, data_path=data_path, seq_len=seq_len, data_percentage=data_percentage,
                         batch_size=batch_size, train_size=train_size, val_size=val_size, test_size=test_size,
                         num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        self.filename = filename  # The specific file to use
        if self.filename is None:
            raise ValueError("Filename must be specified in the config under data_params.")

    def setup(self, stage: Optional[str] = None) -> None:
        file_path = os.path.join(self.data_path, self.filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Extract information from filename
        pattern = r"^\d+_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$"
        match = re.match(pattern, self.filename)
        if not match:
            raise ValueError(f"Filename {self.filename} does not match the expected pattern.")
        mnemonic_name = match.group(1)
        train_end_idx = int(match.group(2))
        anomaly_start_idx = int(match.group(3))
        anomaly_end_idx = int(match.group(4))

        # Load data
        data = np.loadtxt(file_path)
        data = np.nan_to_num(data)  # Handle NaN values

        # Create labels
        labels = np.zeros(len(data), dtype=int)
        labels[anomaly_start_idx - 1:anomaly_end_idx] = 1  # Indices are 0-based

        # Split data into train and test
        train_data = data[:train_end_idx]
        train_labels = labels[:train_end_idx]  # Should be zeros

        test_data = data[train_end_idx:]
        test_labels = labels[train_end_idx:]

        # Normalize data using StandardScaler
        scaler = StandardScaler()
        if len(train_data) > 0:
            train_data = train_data.reshape(-1, 1)
            scaler.fit(train_data)
            train_data = scaler.transform(train_data).flatten()
        if len(test_data) > 0:
            test_data = test_data.reshape(-1, 1)
            test_data = scaler.transform(test_data).flatten()

        # Create datasets
        if len(train_data) >= self.seq_len:
            self.train_dataset = UCRDataset(train_data, train_labels, seq_len=self.seq_len)
        else:
            print(
                f"Training data is shorter than seq_len ({len(train_data)} < {self.seq_len}). Skipping training dataset.")
            self.train_dataset = None

        # Since we are not splitting into validation, set val_dataset to None
        self.val_dataset = None

        if len(test_data) >= self.seq_len:
            self.test_dataset = UCRDataset(test_data, test_labels, seq_len=self.seq_len)
        else:
            print(f"Test data is shorter than seq_len ({len(test_data)} < {self.seq_len}). Skipping test dataset.")
            self.test_dataset = None

        # Log dataset sizes
        if self.train_dataset:
            print(f"Total training sequences: {len(self.train_dataset)}")
        if self.test_dataset:
            print(f"Total test sequences: {len(self.test_dataset)}")