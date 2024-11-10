import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class SMDDataset(Dataset):
    def __init__(self, data_list, labels_list, seq_len=200, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []
        self.labels = []
        self.ts_ids = []  # Time series IDs

        # Iterate over each time series in the data list
        for ts_id, (data, labels) in enumerate(zip(data_list, labels_list)):
            data = torch.tensor(data).float()  # Shape: [num_samples, num_features]
            labels = torch.tensor(labels).float()  # Shape: [num_samples]
            seqs, lbls = self._create_sequences(data, labels)
            self.sequences.extend(seqs)
            self.labels.extend(lbls)
            self.ts_ids.extend([ts_id] * len(seqs))  # Assign ts_id to sequences

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
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.sequences[idx]  # Shape: [seq_len, num_features]
        target = self.labels[idx]
        ts_id = self.ts_ids[idx]
        return {'signal': signal, 'target': target, 'ts_id': ts_id}


class SMDDataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Directories for train, test, and test labels
        train_dir = os.path.join(self.data_path, 'train')
        test_dir = os.path.join(self.data_path, 'test')
        test_label_dir = os.path.join(self.data_path, 'test_label')

        # Get list of files in train and test directories
        train_files = sorted([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))])
        test_files = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
        test_label_files = sorted([f for f in os.listdir(test_label_dir) if os.path.isfile(os.path.join(test_label_dir, f))])

        # Ensure that test files and test label files match
        if set(test_files) != set(test_label_files):
            raise ValueError("Mismatch between test files and test label files.")

        # Initialize lists to hold training data and labels per time series
        train_data_list = []
        train_labels_list = []

        # Load all training files
        for filename in train_files:
            file_path = os.path.join(train_dir, filename)
            data = self.load_file(file_path)
            if data is not None:
                # Training data is assumed to be normal (labels = 0)
                labels = np.zeros(len(data), dtype=int)
                train_data_list.append(data)
                train_labels_list.append(labels)

        # Initialize lists to hold test data and labels per time series
        test_data_list = []
        test_labels_list = []

        # Load all test files and their corresponding labels
        for filename in test_files:
            data_file_path = os.path.join(test_dir, filename)
            label_file_path = os.path.join(test_label_dir, filename)

            data = self.load_file(data_file_path)
            labels = self.load_file(label_file_path, labels=True)
            if data is not None and labels is not None:
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
            Log.error("No training data found.")
            self.train_dataset = None

        # Normalize test data using the same scaler
        if test_data_list:
            for idx in range(len(test_data_list)):
                test_data_list[idx] = scaler.transform(test_data_list[idx])
        else:
            Log.error("No test data found.")
            self.test_dataset = None

        # Apply data percentage filter
        if self.data_percentage < 100:
            num_train_samples = [int(len(data) * (self.data_percentage / 100)) for data in train_data_list]
            train_data_list = [data[:n] for data, n in zip(train_data_list, num_train_samples)]
            train_labels_list = [labels[:n] for labels, n in zip(train_labels_list, num_train_samples)]

            num_test_samples = [int(len(data) * (self.data_percentage / 100)) for data in test_data_list]
            test_data_list = [data[:n] for data, n in zip(test_data_list, num_test_samples)]
            test_labels_list = [labels[:n] for labels, n in zip(test_labels_list, num_test_samples)]

        # Split train data into training and validation sets per time series
        x_train_list = []
        y_train_list = []
        x_val_list = []
        y_val_list = []
        for data, labels in zip(train_data_list, train_labels_list):
            val_size = int(len(data) * (self.val_size / 100))
            if val_size == 0:
                Log.warning("Validation size is zero for one of the time series. Adjust 'val_size' parameter.")
                x_train_list.append(data)
                y_train_list.append(labels)
                # Do not append empty arrays to x_val_list and y_val_list
                # x_val_list.append(np.array([]))
                # y_val_list.append(np.array([]))
            else:
                x_train_list.append(data[:-val_size])
                y_train_list.append(labels[:-val_size])
                x_val_list.append(data[-val_size:])
                y_val_list.append(labels[-val_size:])

        # Create training dataset
        if x_train_list:
            self.train_dataset = SMDDataset(x_train_list, y_train_list, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                Log.error("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            self.train_dataset = None

        # Create validation dataset only if validation data is available
        has_validation_data = any(len(data) >= self.seq_len for data in x_val_list)
        if has_validation_data:
            self.val_dataset = SMDDataset(x_val_list, y_val_list, seq_len=self.seq_len)
            if len(self.val_dataset) == 0:
                Log.warning("No sequences created for validation dataset.")
                self.val_dataset = None
        else:
            Log.warning("Validation data is too short or not available. Skipping validation dataset.")
            self.val_dataset = None

        # Create test dataset
        if test_data_list:
            self.test_dataset = SMDDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            if len(self.test_dataset) == 0:
                Log.error("No sequences created for test dataset.")
                self.test_dataset = None
        else:
            self.test_dataset = None

        # Log dataset sizes
        if self.train_dataset:
            Log.info(f"Total training sequences: {len(self.train_dataset)}")
        else:
            Log.error("Training dataset is empty.")
        if self.val_dataset:
            Log.info(f"Total validation sequences: {len(self.val_dataset)}")
        else:
            Log.warning("Validation dataset is empty.")
        if self.test_dataset:
            Log.info(f"Total test sequences: {len(self.test_dataset)}")
        else:
            Log.error("Test dataset is empty.")

    def load_file(self, file_path, labels=False):
        _, ext = os.path.splitext(file_path)
        try:
            if ext in ['.csv', '.txt']:
                # Use pandas to read CSV and TXT files with comma delimiter
                data = pd.read_csv(file_path, dtype=np.float32, header=None, delimiter=',').values
            elif ext == '.npy':
                data = np.load(file_path)
            else:
                Log.error(f"Unsupported file format: {ext}")
                return None

            # Handle NaN values
            data = np.nan_to_num(data)

            if labels:
                # Ensure labels are integers
                data = data.flatten().astype(int)
            return data
        except Exception as e:
            Log.error(f"Error loading file {file_path}: {e}")
            return None