import os
from typing import Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from nianetvae.dataloaders import BaseDataLoader

#TODO https://chatgpt.com/share/66e5e480-273c-8002-9bf5-8028e4ccc8a8
# Custom dataset class for Yahoo A1Benchmark
class YahooA1Dataset(Dataset):
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


# Custom Yahoo DataLoader for A1Benchmark
class YahooA1DataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        # Load the A1Benchmark dataset
        all_files = [f for f in os.listdir(os.path.join(self.data_path, 'A1Benchmark')) if f.endswith('.csv')]
        all_data = []

        for file in all_files:
            file_path = os.path.join(self.data_path, 'A1Benchmark', file)
            df = pd.read_csv(file_path, header=None, names=['timestamp', 'value', 'is_anomaly'])
            # Skipp the first row, as it contains the column names
            df = df.iloc[1:]
            all_data.append(df)

        combined_df = pd.concat(all_data)
        #TODO make dynamic approach
        # Very dangerours to do this in general since this is the timeseriesdataset which needs temproal dependencies
        combined_df = combined_df.sample(frac = 1, random_state=42)

        # Apply data percentage filter
        combined_df = combined_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        # Split into features (value) and targets (is_anomaly)
        combined_data = pd.to_numeric(combined_df['value']).values
        combined_target = pd.to_numeric(combined_df['is_anomaly']).values

        # Split into train, validation, and test sets
        x_train_val, x_test, y_train_val, y_test = train_test_split(combined_data, combined_target,
                                                                    test_size=self.test_size / 100, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=self.val_size / 100,
                                                          random_state=42)

        # Create datasets
        self.train_dataset = YahooA1Dataset(x_train, y_train)
        self.val_dataset = YahooA1Dataset(x_val, y_val)
        self.test_dataset = YahooA1Dataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)
