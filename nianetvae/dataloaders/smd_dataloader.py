import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from nianetvae.dataloaders import BaseDataLoader


# Inherit from MSLDataset and adjust as needed
class SMDDataset(Dataset):
    def __init__(self, data, targets, seq_len=200, stride=1):
        self.data = torch.tensor(data).float()  # Shape: [num_samples, num_features]
        self.targets = torch.tensor(targets).float()  # Shape: [num_samples]
        self.seq_len = seq_len  # Sequence length for each sample
        self.stride = stride  # Stride for creating sequences
        self.sequences, self.labels = self._create_sequences()  # Generate sequences and corresponding labels

    def _create_sequences(self):
        if len(self.data) < self.seq_len:
            print(
                f"Data length ({len(self.data)}) is less than seq_len ({self.seq_len}). No sequences will be created.")
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
            return torch.empty((0, self.seq_len, self.data.shape[1])), torch.empty((0,))

        return torch.stack(sequences), torch.tensor(seq_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.sequences[idx]  # Shape: [seq_len, num_features]
        target = self.labels[idx].int()
        return {'signal': signal, 'target': target}


class SMDDataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Dynamically get filenames from the data_path
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

        # Initialize lists to hold training data and labels
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

        # Concatenate all loaded training data and labels
        if train_data_list:
            train_data = np.concatenate(train_data_list, axis=0)
            train_labels = np.concatenate(train_labels_list, axis=0)
        else:
            print("No training data found.")
            train_data = np.array([])
            train_labels = np.array([])

        # Initialize lists to hold test data and labels
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

        # Concatenate all loaded test data and labels
        if test_data_list:
            test_data = np.concatenate(test_data_list, axis=0)
            test_labels = np.concatenate(test_labels_list, axis=0)
        else:
            print("No test data found.")
            test_data = np.array([])
            test_labels = np.array([])

        # Normalize train and test data using StandardScaler
        scaler = StandardScaler()
        if train_data.size > 0:
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
        if test_data.size > 0:
            test_data = scaler.transform(test_data)

        # Apply data percentage filter
        num_train_samples = int(len(train_data) * (self.data_percentage / 100))
        num_test_samples = int(len(test_data) * (self.data_percentage / 100))
        train_data = train_data[:num_train_samples]
        train_labels = train_labels[:num_train_samples]
        test_data = test_data[:num_test_samples]
        test_labels = test_labels[:num_test_samples]

        # Split train data into training and validation sets
        val_size = int(len(train_data) * (self.val_size / 100))
        if val_size == 0:
            print("Validation size is zero. Adjust 'val_size' parameter.")
            x_train, x_val = train_data, np.array([])
            y_train, y_val = train_labels, np.array([])
        else:
            x_train, x_val = train_data[:-val_size], train_data[-val_size:]
            y_train, y_val = train_labels[:-val_size], train_labels[-val_size:]

        # Create datasets
        if len(x_train) >= self.seq_len:
            self.train_dataset = SMDDataset(x_train, y_train, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                print("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            print(
                f"Training data is shorter than seq_len ({len(x_train)} < {self.seq_len}). Skipping training dataset.")
            self.train_dataset = None

        if len(x_val) >= self.seq_len:
            self.val_dataset = SMDDataset(x_val, y_val, seq_len=self.seq_len)
            if len(self.val_dataset) == 0:
                print("No sequences created for validation dataset.")
                self.val_dataset = None
        else:
            print(
                f"Validation data is shorter than seq_len ({len(x_val)} < {self.seq_len}). Skipping validation dataset.")
            self.val_dataset = None

        if len(test_data) >= self.seq_len:
            self.test_dataset = SMDDataset(test_data, test_labels, seq_len=self.seq_len)
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

    def load_file(self, file_path, labels=False):
        _, ext = os.path.splitext(file_path)
        try:
            if ext in ['.csv', '.txt']:
                # Use pandas to read CSV and TXT files with comma delimiter
                data = pd.read_csv(file_path, dtype=np.float32, header=None, delimiter=',').values
            elif ext == '.npy':
                data = np.load(file_path)
            else:
                print(f"Unsupported file format: {ext}")
                return None

            # Handle NaN values
            data = np.nan_to_num(data)

            if labels:
                # Ensure labels are integers
                data = data.flatten().astype(int)
            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
