import os
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class KPIDataset(Dataset):
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
        if signal.dim() == 1:
            signal = signal.unsqueeze(-1)
        target = self.labels[idx]
        ts_id = self.ts_ids[idx]
        return {'signal': signal, 'target': target.int(), 'ts_id': ts_id}


class KPIDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list, train_labels_list, test_data_list, test_labels_list = self._load_data_files()

        train_data_list, train_labels_list = self._normalize_and_filter(train_data_list, train_labels_list, fit=True)
        test_data_list, test_labels_list = self._normalize_and_filter(test_data_list, test_labels_list, fit=False)

        self._split_train_validation(train_data_list, train_labels_list)

        if test_data_list:
            self.test_dataset = KPIDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            print(f"Total test sequences: {len(self.test_dataset)}")
        else:
            self.test_dataset = None
            Log.error("Test dataset is empty.")

    def _load_raw_KPI(self, train_filename: str, test_filename: str) -> Tuple[Dict, Dict, Dict, Dict]:
        train_data = pd.read_csv(train_filename)
        train_data = train_data.set_index(['KPI ID', 'timestamp']).sort_index()
        x_train, y_train, scaler = {}, {}, {}
        for name, df in train_data.groupby(level=0):
            x_train[name] = df['value'].to_numpy()
            y_train[name] = df['label'].to_numpy()
            meanv, stdv = df['value'].mean(), df['value'].std()
            scaler[name] = (meanv, stdv)
            x_train[name] = (x_train[name] - meanv) / stdv

        test_data = pd.read_hdf(test_filename)
        test_data['KPI ID'] = test_data['KPI ID'].apply(str)
        test_data = test_data.set_index(['KPI ID', 'timestamp']).sort_index()
        x_test, y_test = {}, {}
        for name, df in test_data.groupby(level=0):
            x_test[name] = df['value'].to_numpy()
            y_test[name] = df['label'].to_numpy()
            x_test[name] = (x_test[name] - scaler[name][0]) / scaler[name][1]

        return x_train, y_train, x_test, y_test

    def _load_data_files(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        train_file = os.path.join(self.data_path, 'train', 'phase2_train.csv')
        test_file = os.path.join(self.data_path, 'test', 'phase2_ground_truth.hdf')

        train_data_dict, train_labels_dict, test_data_dict, test_labels_dict = self._load_raw_KPI(train_file, test_file)

        return (list(train_data_dict.values()), list(train_labels_dict.values()),
                list(test_data_dict.values()), list(test_labels_dict.values()))

    def _normalize_and_filter(self, data_list: List[np.ndarray], labels_list: List[np.ndarray], fit=True) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        if not data_list:
            return [], []

        if fit:
            self.scaler = StandardScaler()  # Set as an attribute of the class
            all_data = np.concatenate(data_list).reshape(-1, 1)
            self.scaler.fit(all_data)
        else:
            if not hasattr(self, 'scaler'):
                raise AttributeError("Scaler not initialized. Ensure training data is normalized before test data.")

        data_list = [self.scaler.transform(data.reshape(-1, 1)).flatten() for data in data_list]

        if self.data_percentage < 100:
            num_samples = [int(len(data) * (self.data_percentage / 100)) for data in data_list]
            data_list = [data[:n] for data, n in zip(data_list, num_samples)]
            labels_list = [labels[:n] for labels, n in zip(labels_list, num_samples)]

        return data_list, labels_list

    def _split_train_validation(self, train_data_list: List[np.ndarray], train_labels_list: List[np.ndarray]) -> None:
        if self.val_size > 0:
            train_data_split, val_data_split, train_labels_split, val_labels_split = [], [], [], []
            for data, labels in zip(train_data_list, train_labels_list):
                data_train, data_val, labels_train, labels_val = train_test_split(
                    data, labels, test_size=self.val_size / 100, random_state=42)
                train_data_split.append(data_train)
                val_data_split.append(data_val)
                train_labels_split.append(labels_train)
                val_labels_split.append(labels_val)

            if train_data_split:
                self.train_dataset = KPIDataset(train_data_split, train_labels_split, seq_len=self.seq_len)
                Log.info(f"Total training sequences: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
                Log.warning("Training dataset is empty.")

            if val_data_split:
                self.val_dataset = KPIDataset(val_data_split, val_labels_split, seq_len=self.seq_len)
                Log.info(f"Total validation sequences: {len(self.val_dataset)}")
            else:
                self.val_dataset = None
                Log.warning("Validation dataset is not created.")
        else:
            self.train_dataset = KPIDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            self.val_dataset = None
            Log.info("Validation dataset is not created as val_size is set to 0.")
