import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class ECG5000Dataset(Dataset):
    def __init__(self, data, targets):
        # Convert data and targets to PyTorch tensors
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()
        # Each sample is already a sequence of length 140

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample and target at the specified index
        signal = self.data[idx]  # Shape: [seq_len]
        # Reshape data if necessary
        if signal.dim() == 1:
            # Univariate data: add an extra dimension to match model input expectations
            signal = signal.unsqueeze(-1)  # Shape: [seq_len, 1]

        target = self.targets[idx]
        return {'signal': signal, 'target': target.int()}


class ECG5000DataLoader(BaseDataLoader):

    def setup(self, stage: Optional[str] = None) -> None:
        # Load training data from ARFF file
        with open(os.path.join(self.data_path, 'ECG5000_TRAIN.arff')) as f:
            train_data, train_meta = arff.loadarff(f)
        # Load testing data from ARFF file
        with open(os.path.join(self.data_path, 'ECG5000_TEST.arff')) as f:
            test_data, test_meta = arff.loadarff(f)

        # Convert the loaded data to pandas DataFrames
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Combine the train and test datasets for splitting
        combined_df = pd.concat([train_df, test_df])

        # Apply data percentage filter to use a subset of the data
        combined_df = combined_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        # Separate features and targets
        combined_data = combined_df.drop(columns=['target']).values  # Shape: [num_samples, seq_len]
        combined_target = pd.to_numeric(combined_df['target']).values  # Original labels

        # Map multiclass labels to binary labels: 0 for normal, 1 for anomaly
        # In the ECG5000 dataset, class '1' is normal, and classes '2' to '5' are anomalies
        combined_target = np.where(combined_target == 1, 0, 1)

        # Do NOT flatten the data; each sample is a sequence of length 140
        # combined_data = combined_data.reshape(-1)

        # Calculate test size and validation size as proportions
        test_size = self.test_size / 100.0  # Proportion of data to be used for testing
        val_size = self.val_size / (100.0 - self.test_size)  # Adjust validation size based on remaining data

        # Split combined data into training+validation and test sets using stratified splitting
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            combined_data, combined_target, test_size=test_size, random_state=42, stratify=combined_target
        )

        # Split training+validation data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
        )

        # Create datasets for training, validation, and testing
        self.train_dataset = ECG5000Dataset(x_train, y_train)
        self.val_dataset = ECG5000Dataset(x_val, y_val)
        self.test_dataset = ECG5000Dataset(x_test, y_test)

        # Log the sizes of the datasets
        Log.info(f"Train size: {len(self.train_dataset)}")
        Log.info(f"Validation size: {len(self.val_dataset)}")
        Log.info(f"Test size: {len(self.test_dataset)}")

        # Log the distribution of classes (normal vs. anomaly) in each set
        train_counts = pd.Series(y_train).value_counts()
        val_counts = pd.Series(y_val).value_counts()
        test_counts = pd.Series(y_test).value_counts()
        Log.info(f"Train class distribution: {train_counts.to_dict()}")
        Log.info(f"Validation class distribution: {val_counts.to_dict()}")
        Log.info(f"Test class distribution: {test_counts.to_dict()}")