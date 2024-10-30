import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset

from nianetvae.dataloaders import BaseDataLoader


# Custom dataset class for Yahoo A1
class YahooA1Dataset(Dataset):
    def __init__(self, data, targets, seq_len=200, stride=1):
        # Convert data and targets to PyTorch tensors
        self.data = torch.tensor(data).float()  # Shape: [num_samples]
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
            return torch.empty((0, self.seq_len)), torch.empty((0,))

        # Initialize lists to hold sequences and labels
        sequences = []
        seq_labels = []
        # Create sequences using a sliding window approach
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            # Extract a sequence of length seq_len
            sequence = self.data[i:i + self.seq_len]  # Shape: [seq_len]
            # Determine the label for the sequence
            # Label is 1 if any of the targets within the sequence indicate an anomaly
            label = 1 if self.targets[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)

        if not sequences:
            # Handle case where no sequences were created
            print(f"No sequences were created for data length {len(self.data)} with seq_len {self.seq_len}.")
            # Return empty tensors if no sequences were created
            return torch.empty((0, self.seq_len)), torch.empty((0,))

        # Stack sequences and labels into tensors
        return torch.stack(sequences), torch.tensor(seq_labels)

    def __len__(self):
        # Return the number of sequences
        return len(self.sequences)

    def __getitem__(self, idx):
        # Get the sequence and label at the specified index
        signal = self.sequences[idx]  # Shape: [seq_len]
        # Reshape data if necessary
        if signal.dim() == 1:
            # Univariate data: add an extra dimension to match model input expectations
            signal = signal.unsqueeze(-1)  # Shape: [seq_len, 1]

        target = self.labels[idx]
        return {'signal': signal, 'target': target.int()}


# Custom DataLoader for Yahoo A1 dataset
class YahooA1DataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Load the A1Benchmark dataset directory
        dataset_dir = os.path.join(self.data_path, 'A1Benchmark')
        # List all CSV files in the dataset directory
        all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        # Lists to hold datasets for each split
        train_datasets = []
        val_datasets = []
        test_datasets = []

        # Iterate over all CSV files in the dataset
        for file in all_files:
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_csv(file_path)

            # Ensure the data is sorted by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Apply data percentage filter (if applicable)
            if self.data_percentage < 100.0:
                df = df.iloc[:int(len(df) * self.data_percentage / 100.0)]

            # Extract 'value' and 'is_anomaly' columns
            data = pd.to_numeric(df['value']).values  # Shape: [num_samples]
            target = pd.to_numeric(df['is_anomaly']).values  # Shape: [num_samples]

            # Skip sequences shorter than seq_len
            if len(data) < self.seq_len:
                print(f"Sequence in file {file} is shorter than seq_len ({len(data)} < {self.seq_len}). Skipping.")
                continue

            # Split into train, validation, and test sets using user-defined sizes
            total_length = len(data)

            # Convert percentages to proportions
            train_prop = self.train_size / 100.0
            val_prop = self.val_size / 100.0
            test_prop = self.test_size / 100.0

            # Ensure that proportions sum to 1.0
            total_prop = train_prop + val_prop + test_prop
            if not np.isclose(total_prop, 1.0):
                # Normalize proportions if they don't sum to 1
                train_prop /= total_prop
                val_prop /= total_prop
                test_prop /= total_prop

            # Calculate indices for splitting the data
            train_end = int(train_prop * total_length)
            val_end = int((train_prop + val_prop) * total_length)

            # Split the data and targets into train, val, and test sets
            x_train = data[:train_end]
            y_train = target[:train_end]
            x_val = data[train_end:val_end]
            y_val = target[train_end:val_end]
            x_test = data[val_end:]
            y_test = target[val_end:]

            # Skip splits that are shorter than seq_len
            datasets = [
                ('train', x_train, y_train, train_datasets),
                ('val', x_val, y_val, val_datasets),
                ('test', x_test, y_test, test_datasets),
            ]

            for split_name, x_split, y_split, dataset_list in datasets:
                if len(x_split) < self.seq_len:
                    print(
                        f"{split_name.capitalize()} split in file {file} is shorter than seq_len ({len(x_split)} < {self.seq_len}). Skipping.")
                    continue

                # Create a dataset for the current split
                dataset = YahooA1Dataset(x_split, y_split, seq_len=self.seq_len)
                if len(dataset) > 0:
                    # Add the dataset to the corresponding list
                    dataset_list.append(dataset)
                else:
                    print(f"No sequences created for {split_name} split in file {file}. Skipping.")

        # Combine datasets from all files using ConcatDataset
        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
        else:
            self.train_dataset = None
            print("No training data available.")

        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
        else:
            self.val_dataset = None
            print("No validation data available.")

        if test_datasets:
            self.test_dataset = ConcatDataset(test_datasets)
        else:
            self.test_dataset = None
            print("No test data available.")

        # Log dataset sizes
        if self.train_dataset:
            print(f"Total training sequences: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Total validation sequences: {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"Total test sequences: {len(self.test_dataset)}")
