import os
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class YahooA1Dataset(Dataset):
    def __init__(self, data_list: List[np.ndarray], labels_list: List[np.ndarray], seq_len=200, stride=1):
        self.seq_len = seq_len
        self.stride = stride
        self.sequences, self.labels, self.ts_ids = [], [], []

        for ts_id, (data, labels) in enumerate(zip(data_list, labels_list)):
            data = torch.tensor(data).float()
            labels = torch.tensor(labels).float()
            seqs, lbls = self._create_sequences(data, labels)
            self.sequences.extend(seqs)
            self.labels.extend(lbls)
            self.ts_ids.extend([ts_id] * len(seqs))

        if self.sequences:
            self.sequences = torch.stack(self.sequences)
            self.labels = torch.tensor(self.labels).int()
            self.ts_ids = torch.tensor(self.ts_ids).int()
        else:
            self.sequences = torch.empty((0, self.seq_len))
            self.labels = torch.empty((0,), dtype=torch.int)
            self.ts_ids = torch.empty((0,), dtype=torch.int)

    def _create_sequences(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        sequences, seq_labels = [], []
        for i in range(0, len(data) - self.seq_len + 1, self.stride):
            sequence = data[i:i + self.seq_len]
            label = 1 if labels[i:i + self.seq_len].sum() > 0 else 0
            sequences.append(sequence)
            seq_labels.append(label)
        return sequences, seq_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        signal = self.sequences[idx]
        target = self.labels[idx]
        ts_id = self.ts_ids[idx]
        return {'signal': signal.unsqueeze(-1), 'target': target, 'ts_id': ts_id}


class YahooA1DataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list, train_labels_list, val_data_list, val_labels_list, test_data_list, test_labels_list = self._load_data_files()

        train_data_list, train_labels_list = self._normalize_and_filter(train_data_list, train_labels_list, fit=True)
        val_data_list, val_labels_list = self._normalize_and_filter(val_data_list, val_labels_list, fit=False)
        test_data_list, test_labels_list = self._normalize_and_filter(test_data_list, test_labels_list, fit=False)

        self.train_dataset = YahooA1Dataset(train_data_list, train_labels_list, seq_len=self.seq_len)
        self.val_dataset = YahooA1Dataset(val_data_list, val_labels_list, seq_len=self.seq_len)
        self.test_dataset = YahooA1Dataset(test_data_list, test_labels_list, seq_len=self.seq_len)

        # Log dataset sizes
        if self.train_dataset:
            Log.info(f"Total training sequences: {len(self.train_dataset)}")
        else:
            Log.warning("Training dataset is empty.")
        if self.val_dataset and len(self.val_dataset) > 0:
            Log.info(f"Total validation sequences: {len(self.val_dataset)}")
        else:
            Log.warning("Validation dataset is empty.")
        if self.test_dataset:
            Log.info(f"Total test sequences: {len(self.test_dataset)}")
        else:
            Log.warning("Test dataset is empty.")

    def _load_data_files(self) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        dataset_dir = os.path.join(self.data_path, 'A1Benchmark')
        all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        train_data_list, train_labels_list, val_data_list, val_labels_list, test_data_list, test_labels_list = [], [], [], [], [], []

        for file in all_files:
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_csv(file_path)
            df = df.sort_values('timestamp').reset_index(drop=True)

            if self.data_percentage < 100.0:
                df = df.iloc[:int(len(df) * self.data_percentage / 100.0)]

            data = pd.to_numeric(df['value']).values
            target = pd.to_numeric(df['is_anomaly']).values

            if len(data) < self.seq_len:
                Log.warning(f"File {file} is too short for sequence length {self.seq_len}. Skipping.")
                continue

            total_length = len(data)
            test_size = 40
            val_prop = self.val_size / 100.0
            train_prop = (100 - self.val_size - test_size) / 100.0

            train_end = int(train_prop * total_length)
            val_end = int((train_prop + val_prop) * total_length)

            x_train = data[:train_end]
            y_train = target[:train_end]
            x_val = data[train_end:val_end]
            y_val = target[train_end:val_end]
            x_test = data[val_end:]
            y_test = target[val_end:]

            train_data_list.append(x_train)
            train_labels_list.append(y_train)
            val_data_list.append(x_val)
            val_labels_list.append(y_val)
            test_data_list.append(x_test)
            test_labels_list.append(y_test)

        return train_data_list, train_labels_list, val_data_list, val_labels_list, test_data_list, test_labels_list

    def _normalize_and_filter(self, data_list: List[np.ndarray], labels_list: List[np.ndarray], fit=True) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        if not data_list:
            return [], []

        if fit:
            self.scaler = StandardScaler()
            all_data = np.concatenate(data_list).reshape(-1, 1)
            self.scaler.fit(all_data)

        data_list = [self.scaler.transform(data.reshape(-1, 1)).flatten() for data in data_list]

        if self.data_percentage < 100:
            num_samples = [int(len(data) * (self.data_percentage / 100)) for data in data_list]
            data_list = [data[:n] for data, n in zip(data_list, num_samples)]
            labels_list = [labels[:n] for labels, n in zip(labels_list, num_samples)]

        return data_list, labels_list
