import os
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from nianetvae.dataloaders import BaseDataLoader


# Custom dataset class for KPI
class KPIDataset(Dataset):
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
        sample = {'signal': self.data[idx], 'target': self.targets[idx].int()}
        return sample


# Custom KPI DataLoader
class KPIDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        # Load train and test data from respective CSV files
        train_file = os.path.join(self.data_path, 'train.csv')
        test_file = os.path.join(self.data_path, 'test.csv')

        # Load train.csv and test.csv into dataframes
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # Only keep 'timestamp' and 'value' columns for data and 'KPI ID' for identification if needed
        train_df = train_df[['timestamp', 'value']]  # Keeping 'timestamp' in case it's useful later
        test_df = test_df[['timestamp', 'value']]

        # Apply data percentage filter
        train_df = train_df.sample(frac=self.data_percentage / 100.0, random_state=42)
        test_df = test_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        # Split into features (value) and targets (label) for both train and test
        train_values = pd.to_numeric(train_df['value']).values
        test_values = pd.to_numeric(test_df['value']).values

        # There are no labels in the dataset based on the description, so labels can be zeros
        train_labels = [0] * len(train_values)  # You can modify this based on the actual labels
        test_labels = [0] * len(test_values)

        # Handle NaN values
        train_values = pd.Series(train_values).fillna(0).values  # Replace NaN with 0
        test_values = pd.Series(test_values).fillna(0).values  # Replace NaN with 0

        # Normalize train and test data using the training data's mean and std
        mean_data, std_data = train_values.mean(), train_values.std()
        train_values = (train_values - mean_data) / (std_data if std_data != 0 else 1)
        test_values = (test_values - mean_data) / (std_data if std_data != 0 else 1)

        # Split train data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(train_values, train_labels, test_size=self.val_size / 100,
                                                          random_state=42)

        # Create datasets
        self.train_dataset = KPIDataset(x_train, y_train)
        self.val_dataset = KPIDataset(x_val, y_val)
        self.test_dataset = KPIDataset(test_values, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)
