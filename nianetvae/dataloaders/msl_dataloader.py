import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from nianetvae.dataloaders import BaseDataLoader


# Custom dataset class for MSL
class MSLDataset(Dataset):
    def __init__(self, data, targets, window_size=200, stride=1):
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()
        self.window_size = window_size
        self.stride = stride
        self.data, self.targets = self._window_data()

    def _window_data(self):
        # Create sliding windows for the data, with corresponding labels
        windows = []
        wlabels = []
        for i in range(0, len(self.data) - self.window_size, self.stride):
            window = self.data[i:i + self.window_size]
            label = 1 if self.targets[i:i + self.window_size].sum() > 0 else 0
            windows.append(window)
            wlabels.append(label)
        return torch.stack(windows), torch.tensor(wlabels)

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


# Custom MSL DataLoader with CSV anomaly reading
class MSLDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        wsz, stride = 200, 1

        # Load anomaly sequences from labeled_anomalies.csv
        anomaly_file = os.path.join(self.data_path, 'labeled_anomalies.csv')
        anomaly_info = pd.read_csv(anomaly_file)

        # Filter out MSL-specific rows
        msl_anomalies = anomaly_info[anomaly_info['spacecraft'] == 'MSL']

        # Load all MSL .npy files based on chan_id in the CSV
        train_data = []
        train_labels = []
        for index, row in msl_anomalies.iterrows():
            file_name = row['chan_id'] + '.npy'
            file_path = os.path.join(self.data_path, 'train', file_name)
            if os.path.exists(file_path):
                data = np.load(file_path)

                # Handle NaN values
                data = np.nan_to_num(data)

                # Create anomaly labels based on anomaly_sequences
                labels = np.zeros(len(data), dtype=bool)
                anomaly_sequences = eval(row['anomaly_sequences'])
                for anomaly_range in anomaly_sequences:
                    labels[anomaly_range[0]:anomaly_range[1] + 1] = True

                train_data.append(data)
                train_labels.append(labels)

        # Concatenate all loaded train data and labels
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # Apply the same for the test data if necessary
        test_data = []
        test_labels = []
        for index, row in msl_anomalies.iterrows():
            file_name = row['chan_id'] + '.npy'
            file_path = os.path.join(self.data_path, 'test', file_name)
            if os.path.exists(file_path):
                data = np.load(file_path)

                # Handle NaN values
                data = np.nan_to_num(data)

                # Create anomaly labels for the test data
                labels = np.zeros(len(data), dtype=bool)
                anomaly_sequences = eval(row['anomaly_sequences'])
                for anomaly_range in anomaly_sequences:
                    labels[anomaly_range[0]:anomaly_range[1] + 1] = True

                test_data.append(data)
                test_labels.append(labels)

        # Concatenate test data and labels
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # Normalize train and test data
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Apply data percentage filter (if needed)
        num_train_samples = int(len(train_data) * (self.data_percentage / 100))
        num_test_samples = int(len(test_data) * (self.data_percentage / 100))
        train_data = train_data[:num_train_samples]
        train_labels = train_labels[:num_train_samples]
        test_data = test_data[:num_test_samples]
        test_labels = test_labels[:num_test_samples]

        # Split train data into training and validation sets
        val_size = int(len(train_data) * (self.val_size / 100))
        x_train, x_val = train_data[:-val_size], train_data[-val_size:]
        y_train, y_val = train_labels[:-val_size], train_labels[-val_size:]

        # Create datasets
        self.train_dataset = MSLDataset(x_train, y_train, window_size=wsz, stride=stride)
        self.val_dataset = MSLDataset(x_val, y_val, window_size=wsz, stride=stride)
        self.test_dataset = MSLDataset(test_data, test_labels, window_size=wsz, stride=stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)
