import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from nianetvae.dataloaders import BaseDataLoader


class MSLDataset(Dataset):
    def __init__(self, data_list, labels_list, seq_len=200, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []
        self.labels = []
        self.ts_ids = []  # List to store time series IDs

        # Iterate over each time series in the data list
        for ts_id, (data, labels) in enumerate(zip(data_list, labels_list)):
            data = torch.tensor(data).float()  # Shape: [num_samples, num_features]
            labels = torch.tensor(labels).float()  # Shape: [num_samples]
            seqs, lbls = self._create_sequences(data, labels)
            self.sequences.extend(seqs)
            self.labels.extend(lbls)
            self.ts_ids.extend([ts_id] * len(seqs))  # Assign the same ts_id to all sequences from this time series

        if self.sequences:
            # Stack sequences and labels into tensors
            self.sequences = torch.stack(self.sequences)  # Shape: [num_sequences, seq_len, num_features]
            self.labels = torch.tensor(self.labels).int()  # Shape: [num_sequences]
            self.ts_ids = torch.tensor(self.ts_ids).int()  # Shape: [num_sequences]
        else:
            # Handle the case where no sequences were created
            self.sequences = torch.empty((0, self.seq_len, data_list[0].shape[1]))
            self.labels = torch.empty((0,), dtype=torch.int)
            self.ts_ids = torch.empty((0,), dtype=torch.int)

    def _create_sequences(self, data, labels):
        sequences = []
        seq_labels = []
        num_samples = len(data)

        # Create sequences within the current time series
        for i in range(0, num_samples - self.seq_len + 1, self.stride):
            # Extract a sequence of length seq_len
            sequence = data[i:i + self.seq_len]  # Shape: [seq_len, num_features]
            # Determine the label for the sequence
            # Label is 1 if any of the targets within the sequence indicate an anomaly
            label = 1 if labels[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)

        return sequences, seq_labels

    def __len__(self):
        # Return the number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get the sequence, label, and ts_id at the specified index
        signal = self.sequences[idx]  # Shape: [seq_len, num_features]
        target = self.labels[idx]
        ts_id = self.ts_ids[idx]
        return {'signal': signal, 'target': target, 'ts_id': ts_id}



# Custom MSL DataLoader with separate handling of multiple time series
class MSLDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        # Load anomaly sequences from labeled_anomalies.csv
        anomaly_file = os.path.join(self.data_path, 'labeled_anomalies.csv')
        anomaly_info = pd.read_csv(anomaly_file)

        # Filter out rows for the specified dataset_name (spacecraft)
        spacecraft_anomalies = anomaly_info[anomaly_info['spacecraft'] == self.dataset_name]

        # Initialize lists to hold training data and labels
        train_data_list = []
        train_labels_list = []

        # Load all .npy files based on chan_id in the CSV for training data
        for index, row in spacecraft_anomalies.iterrows():
            file_name = row['chan_id'] + '.npy'
            file_path = os.path.join(self.data_path, 'train', file_name)
            if os.path.exists(file_path):
                # Load the data file
                data = np.load(file_path)  # Shape: [num_samples, num_features]

                # Handle NaN values by replacing them with zeros
                data = np.nan_to_num(data)

                # Create anomaly labels based on anomaly_sequences
                labels = np.zeros(len(data), dtype=int)
                anomaly_sequences = eval(row['anomaly_sequences'])
                for anomaly_range in anomaly_sequences:
                    # Mark anomalies in the labels array
                    labels[anomaly_range[0]:anomaly_range[1] + 1] = 1

                train_data_list.append(data)
                train_labels_list.append(labels)

        # Initialize lists to hold test data and labels
        test_data_list = []
        test_labels_list = []

        # Load all .npy files based on chan_id in the CSV for test data
        for index, row in spacecraft_anomalies.iterrows():
            file_name = row['chan_id'] + '.npy'
            file_path = os.path.join(self.data_path, 'test', file_name)
            if os.path.exists(file_path):
                # Load the data file
                data = np.load(file_path)  # Shape: [num_samples, num_features]

                # Handle NaN values by replacing them with zeros
                data = np.nan_to_num(data)

                # Create anomaly labels for the test data
                labels = np.zeros(len(data), dtype=int)
                anomaly_sequences = eval(row['anomaly_sequences'])
                for anomaly_range in anomaly_sequences:
                    # Mark anomalies in the labels array
                    labels[anomaly_range[0]:anomaly_range[1] + 1] = 1

                test_data_list.append(data)
                test_labels_list.append(labels)

        # Normalize train data
        scaler = StandardScaler()
        if train_data_list:
            # Concatenate all training data to fit the scaler
            all_train_data = np.concatenate(train_data_list, axis=0)
            scaler.fit(all_train_data)
            # Transform each time series individually
            for idx in range(len(train_data_list)):
                train_data_list[idx] = scaler.transform(train_data_list[idx])
        else:
            print("No training data found.")
            self.train_dataset = None

        # Normalize test data using the same scaler
        if test_data_list:
            for idx in range(len(test_data_list)):
                test_data_list[idx] = scaler.transform(test_data_list[idx])
        else:
            print("No test data found.")
            self.test_dataset = None

        # Create training dataset
        if train_data_list:
            self.train_dataset = MSLDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                print("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            self.train_dataset = None

        # No validation dataset
        self.val_dataset = None

        # Create test dataset
        if test_data_list:
            self.test_dataset = MSLDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            if len(self.test_dataset) == 0:
                print("No sequences created for test dataset.")
                self.test_dataset = None
        else:
            self.test_dataset = None

        # Log dataset sizes
        if self.train_dataset:
            print(f"Total training sequences: {len(self.train_dataset)}")
        else:
            print("Training dataset is empty.")
        if self.val_dataset:
            print(f"Total validation sequences: {len(self.val_dataset)}")
        else:
            print("Validation dataset is empty.")
        if self.test_dataset:
            print(f"Total test sequences: {len(self.test_dataset)}")
        else:
            print("Test dataset is empty.")