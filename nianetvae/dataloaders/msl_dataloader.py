import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from nianetvae.dataloaders import BaseDataLoader


# Custom dataset class for MSL and SMAP
class MSLDataset(Dataset):
    def __init__(self, data, targets, seq_len=200, stride=1):
        # Convert data and targets to PyTorch tensors
        self.data = torch.tensor(data).float()  # Shape: [num_samples, num_features]
        self.targets = torch.tensor(targets).float()  # Shape: [num_samples]
        self.seq_len = seq_len  # Sequence length for each sample
        self.stride = stride  # Stride for creating sequences
        self.sequences, self.labels = self._create_sequences()  # Generate sequences and corresponding labels

    def _create_sequences(self):
        # Check if data length is sufficient to create sequences
        if len(self.data) < self.seq_len:
            print(
                f"Data length ({len(self.data)}) is less than seq_len ({self.seq_len}). No sequences will be created.")
            # Return empty tensors if not enough data
            return torch.empty((0, self.seq_len, self.data.shape[1])), torch.empty((0,))

        sequences = []
        seq_labels = []
        # Create sequences using a sliding window approach
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            # Extract a sequence of length seq_len
            sequence = self.data[i:i + self.seq_len]  # Shape: [seq_len, num_features]
            # Determine the label for the sequence
            # Label is 1 if any of the targets within the sequence indicate an anomaly
            label = 1 if self.targets[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)

        if not sequences:
            print(f"No sequences were created for data length {len(self.data)} with seq_len {self.seq_len}.")
            # Return empty tensors if no sequences were created
            return torch.empty((0, self.seq_len, self.data.shape[1])), torch.empty((0,))

        # Stack sequences and labels into tensors
        return torch.stack(sequences), torch.tensor(seq_labels)

    def __len__(self):
        # Return the number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get the sequence and label at the specified index
        signal = self.sequences[idx]  # Shape: [seq_len, num_features]
        target = self.labels[idx]
        return {'signal': signal, 'target': target.int()}


# Custom MSL DataLoader with CSV anomaly reading
class MSLDataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Load anomaly sequences from labeled_anomalies.csv
        anomaly_file = os.path.join(self.data_path, 'labeled_anomalies.csv')
        anomaly_info = pd.read_csv(anomaly_file)

        # Filter out rows for the specified spacecraft
        spacecraft_anomalies = anomaly_info[anomaly_info['spacecraft'] == self.dataset_type]

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

        # Concatenate all loaded training data and labels
        if train_data_list:
            train_data = np.concatenate(train_data_list, axis=0)  # Shape: [total_samples, num_features]
            train_labels = np.concatenate(train_labels_list, axis=0)  # Shape: [total_samples]
        else:
            print("No training data found.")
            train_data = np.array([])
            train_labels = np.array([])

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

        # Concatenate all loaded test data and labels
        if test_data_list:
            test_data = np.concatenate(test_data_list, axis=0)  # Shape: [total_samples, num_features]
            test_labels = np.concatenate(test_labels_list, axis=0)  # Shape: [total_samples]
        else:
            print("No test data found.")
            test_data = np.array([])
            test_labels = np.array([])

        # Normalize train and test data using StandardScaler
        scaler = StandardScaler()
        if train_data.size > 0:
            # Fit the scaler on the training data
            scaler.fit(train_data)
            # Transform the training data
            train_data = scaler.transform(train_data)
        if test_data.size > 0:
            # Transform the test data using the same scaler
            test_data = scaler.transform(test_data)

        # Apply data percentage filter to use a subset of the data
        num_train_samples = int(len(train_data) * (self.data_percentage / 100))
        num_test_samples = int(len(test_data) * (self.data_percentage / 100))
        train_data = train_data[:num_train_samples]
        train_labels = train_labels[:num_train_samples]
        test_data = test_data[:num_test_samples]
        test_labels = test_labels[:num_test_samples]

        # Split train data into training and validation sets based on val_size
        val_size = int(len(train_data) * (self.val_size / 100))
        if val_size == 0:
            print("Validation size is zero. Adjust 'val_size' parameter.")
            x_train, x_val = train_data, np.array([])
            y_train, y_val = train_labels, np.array([])
        else:
            # No shuffling to preserve temporal order
            x_train, x_val = train_data[:-val_size], train_data[-val_size:]
            y_train, y_val = train_labels[:-val_size], train_labels[-val_size:]

        # Create datasets if data length is sufficient
        if len(x_train) >= self.seq_len:
            self.train_dataset = MSLDataset(x_train, y_train, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                print("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            print(
                f"Training data is shorter than seq_len ({len(x_train)} < {self.seq_len}). Skipping training dataset.")
            self.train_dataset = None

        if len(x_val) >= self.seq_len:
            self.val_dataset = MSLDataset(x_val, y_val, seq_len=self.seq_len)
            if len(self.val_dataset) == 0:
                print("No sequences created for validation dataset.")
                self.val_dataset = None
        else:
            print(
                f"Validation data is shorter than seq_len ({len(x_val)} < {self.seq_len}). Skipping validation dataset.")
            self.val_dataset = None

        if len(test_data) >= self.seq_len:
            self.test_dataset = MSLDataset(test_data, test_labels, seq_len=self.seq_len)
            if len(self.test_dataset) == 0:
                print("No sequences created for test dataset.")
                self.test_dataset = None
        else:
            print(f"Test data is shorter than seq_len ({len(test_data)} < {self.seq_len}). Skipping test dataset.")
            self.test_dataset = None

        # Log dataset sizes
        if self.train_dataset:
            print(f"Total training sequences: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Total validation sequences: {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"Total test sequences: {len(self.test_dataset)}")
