import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from nianetvae.dataloaders import BaseDataLoader

# Updated KPIDataset class to handle multiple time series
class KPIDataset(Dataset):
    def __init__(self, data_list, labels_list, seq_len=200, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.sequences = []
        self.labels = []
        self.ts_ids = []  # Time series IDs

        # Iterate over each time series in the data list
        for ts_id, (data, labels) in enumerate(zip(data_list, labels_list)):
            data = torch.tensor(data).float()  # Shape: [num_samples]
            labels = torch.tensor(labels).float()  # Shape: [num_samples]
            seqs, lbls = self._create_sequences(data, labels)
            self.sequences.extend(seqs)
            self.labels.extend(lbls)
            self.ts_ids.extend([ts_id] * len(seqs))  # Assign ts_id to sequences

        if self.sequences:
            # Stack sequences and labels into tensors
            self.sequences = torch.stack(self.sequences)  # Shape: [num_sequences, seq_len]
            self.labels = torch.tensor(self.labels).int()  # Shape: [num_sequences]
            self.ts_ids = torch.tensor(self.ts_ids).int()  # Shape: [num_sequences]
        else:
            # Handle the case where no sequences were created
            self.sequences = torch.empty((0, self.seq_len))
            self.labels = torch.empty((0,), dtype=torch.int)
            self.ts_ids = torch.empty((0,), dtype=torch.int)

    def _create_sequences(self, data, labels):
        sequences = []
        seq_labels = []
        num_samples = len(data)

        # Create sequences within the current time series
        for i in range(0, num_samples - self.seq_len + 1, self.stride):
            # Extract a sequence of length seq_len
            sequence = data[i:i + self.seq_len]  # Shape: [seq_len]
            # Determine the label for the sequence
            label = 1 if labels[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)

        return sequences, seq_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.sequences[idx]  # Shape: [seq_len]
        # Reshape data if necessary
        if signal.dim() == 1:
            # Univariate data: add an extra dimension to match model input expectations
            signal = signal.unsqueeze(-1)  # Shape: [seq_len, 1]
        target = self.labels[idx]
        ts_id = self.ts_ids[idx]
        return {'signal': signal, 'target': target.int(), 'ts_id': ts_id}


# Updated KPI DataLoader class
class KPIDataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Directories for train and test data
        train_dir = os.path.join(self.data_path, 'train')
        test_dir = os.path.join(self.data_path, 'test')

        # Get list of files in train and test directories
        train_files = sorted([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))])
        test_files = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])

        # Initialize lists to hold training data and labels per time series
        train_data_list = []
        train_labels_list = []

        # Load all training files
        for filename in train_files:
            file_path = os.path.join(train_dir, filename)
            data, labels = self.load_file(file_path, train=True)
            if data is not None:
                # Append data and zero labels to lists (unsupervised learning)
                train_data_list.append(data)
                train_labels_list.append(np.zeros(len(data), dtype=int))  # Unsupervised: use zero labels

        # Initialize lists to hold test data and labels per time series
        test_data_list = []
        test_labels_list = []

        # Load all test files
        for filename in test_files:
            file_path = os.path.join(test_dir, filename)
            data, labels = self.load_file(file_path, train=False)
            if data is not None:
                test_data_list.append(data)
                test_labels_list.append(labels)  # Test data contains labels

        # Normalize train data
        scaler = StandardScaler()
        if train_data_list:
            # Concatenate all training data to fit the scaler
            all_train_data = np.concatenate(train_data_list, axis=0).reshape(-1, 1)
            scaler.fit(all_train_data)
            # Transform each time series individually
            for idx in range(len(train_data_list)):
                train_data_list[idx] = scaler.transform(train_data_list[idx].reshape(-1, 1)).flatten()
        else:
            print("No training data found.")
            self.train_dataset = None

        # Normalize test data using the same scaler
        if test_data_list:
            for idx in range(len(test_data_list)):
                test_data_list[idx] = scaler.transform(test_data_list[idx].reshape(-1, 1)).flatten()
        else:
            print("No test data found.")
            self.test_dataset = None

        # Apply data percentage filter
        if self.data_percentage < 100:
            num_train_samples = [int(len(data) * (self.data_percentage / 100)) for data in train_data_list]
            train_data_list = [data[:n] for data, n in zip(train_data_list, num_train_samples)]
            train_labels_list = [labels[:n] for labels, n in zip(train_labels_list, num_train_samples)]

            num_test_samples = [int(len(data) * (self.data_percentage / 100)) for data in test_data_list]
            test_data_list = [data[:n] for data, n in zip(test_data_list, num_test_samples)]
            test_labels_list = [labels[:n] for labels, n in zip(test_labels_list, num_test_samples)]

        # Create datasets
        if train_data_list:
            self.train_dataset = KPIDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                print("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            self.train_dataset = None

        # No validation dataset (or create one if needed)
        self.val_dataset = None

        if test_data_list:
            self.test_dataset = KPIDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
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

    def load_file(self, file_path, train=True):
        try:
            # Load the CSV file into a dataframe
            df = pd.read_csv(file_path)
            # Ensure 'value' and 'label' columns exist for test data
            if 'value' not in df.columns:
                print(f"File {file_path} does not contain 'value' column.")
                return None, None

            # Extract 'value' column and handle NaNs
            values = pd.to_numeric(df['value'], errors='coerce').fillna(0).values

            if not train and 'label' in df.columns:
                labels = df['label'].fillna(0).astype(int).values
            else:
                labels = np.zeros(len(values), dtype=int)  # For unsupervised learning, use zero labels

            return values, labels
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None
