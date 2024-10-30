import os
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from nianetvae.dataloaders import BaseDataLoader


# Custom dataset class for KPI
class KPIDataset(Dataset):
    def __init__(self, data, targets, seq_len=200, stride=1):
        # Convert data and targets to PyTorch tensors
        self.data = torch.tensor(data).float()  # Shape: [num_samples]
        self.targets = torch.tensor(targets).float()  # Shape: [num_samples]
        self.seq_len = seq_len  # Sequence length for each sample
        self.stride = stride  # Stride for creating sequences
        self.data, self.targets = self._create_sequences()  # Generate sequences and corresponding labels

    def _create_sequences(self):
        # Create sliding sequences for the data, with corresponding labels
        sequences = []
        seq_labels = []
        # Iterate over the data to create sequences
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            # Extract a sequence of length seq_len
            sequence = self.data[i:i + self.seq_len]  # Shape: [seq_len]
            # Determine the label for the sequence
            # Since we have no anomaly labels, we default to 0
            label = 1 if self.targets[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)
        if not sequences:
            # Handle case where no sequences were created
            print(f"No sequences created: data length {len(self.data)} is less than seq_len {self.seq_len}")
            # Return empty tensors if no sequences were created
            return torch.empty((0, self.seq_len)), torch.empty((0,))
        # Stack sequences and labels into tensors
        return torch.stack(sequences), torch.tensor(seq_labels)

    def __len__(self):
        # Return the number of sequences
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sequence and label at the specified index
        signal = self.data[idx]  # Shape: [seq_len]
        # Reshape data if necessary
        if signal.dim() == 1:
            # Univariate data: add an extra dimension to match model input expectations
            signal = signal.unsqueeze(-1)  # Shape: [seq_len, 1]
        target = self.targets[idx]
        return {'signal': signal, 'target': target.int()}


# Custom KPI DataLoader
class KPIDataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Load train and test data from respective CSV files
        train_file = os.path.join(self.data_path, 'train.csv')
        test_file = os.path.join(self.data_path, 'test.csv')

        # Load train.csv and test.csv into dataframes
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Only keep 'timestamp' and 'value' columns for data
        train_df = train_df[['timestamp', 'value']]
        test_df = test_df[['timestamp', 'value']]

        # Apply data percentage filter to use a subset of the data
        train_df = train_df.sample(frac=self.data_percentage / 100.0, random_state=42)
        test_df = test_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        # Convert 'value' column to numeric and extract values
        train_values = pd.to_numeric(train_df['value']).values  # Shape: [num_samples]
        test_values = pd.to_numeric(test_df['value']).values  # Shape: [num_samples]

        # Since there are no labels, create zero labels
        train_labels = [0] * len(train_values)
        test_labels = [0] * len(test_values)

        # Handle NaN values by replacing them with 0
        train_values = pd.Series(train_values).fillna(0).values
        test_values = pd.Series(test_values).fillna(0).values

        # Normalize train and test data using the training data's mean and std
        mean_data, std_data = train_values.mean(), train_values.std()
        # Avoid division by zero by checking if std_data is not zero
        train_values = (train_values - mean_data) / (std_data if std_data != 0 else 1)
        test_values = (test_values - mean_data) / (std_data if std_data != 0 else 1)

        # Split train data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            train_values, train_labels, test_size=self.val_size / 100, random_state=42
        )

        # Create datasets if data length is sufficient
        if len(x_train) >= self.seq_len:
            self.train_dataset = KPIDataset(x_train, y_train, seq_len=self.seq_len)
            if len(self.train_dataset) == 0:
                print("No sequences created for training dataset.")
                self.train_dataset = None
        else:
            print(
                f"Training data is shorter than seq_len ({len(x_train)} < {self.seq_len}). Skipping training dataset.")
            self.train_dataset = None

        if len(x_val) >= self.seq_len:
            self.val_dataset = KPIDataset(x_val, y_val, seq_len=self.seq_len)
            if len(self.val_dataset) == 0:
                print("No sequences created for validation dataset.")
                self.val_dataset = None
        else:
            print(
                f"Validation data is shorter than seq_len ({len(x_val)} < {self.seq_len}). Skipping validation dataset.")
            self.val_dataset = None

        if len(test_values) >= self.seq_len:
            self.test_dataset = KPIDataset(test_values, test_labels, seq_len=self.seq_len)
            if len(self.test_dataset) == 0:
                print("No sequences created for test dataset.")
                self.test_dataset = None
        else:
            print(f"Test data is shorter than seq_len ({len(test_values)} < {self.seq_len}). Skipping test dataset.")
            self.test_dataset = None
