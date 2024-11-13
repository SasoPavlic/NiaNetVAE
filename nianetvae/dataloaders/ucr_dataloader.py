import os
import re
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class UCRDataset(Dataset):
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


class UCRDataLoader(BaseDataLoader):
    def __init__(self, dataset_name: str, data_path: str, seq_len: int, data_percentage: float = 100.0,
                 batch_size: int = 32, val_size: float = 10.0,
                 num_workers: int = 0, pin_memory: bool = False, persistent_workers: bool = False,
                 filename: str = None, **kwargs):
        super().__init__(dataset_name=dataset_name, data_path=data_path, seq_len=seq_len,
                         data_percentage=data_percentage,
                         batch_size=batch_size, val_size=val_size, num_workers=num_workers, pin_memory=pin_memory,
                         persistent_workers=persistent_workers)
        self.filename = filename  # The specific file to use
        if self.filename is None:
            raise ValueError("Filename must be specified in the config under data_params.")

    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list, train_labels_list, test_data_list, test_labels_list = self._load_data_files()

        train_data_list, train_labels_list = self._normalize_and_filter(train_data_list, train_labels_list, fit=True)
        test_data_list, test_labels_list = self._normalize_and_filter(test_data_list, test_labels_list, fit=False)

        self._split_train_validation(train_data_list, train_labels_list)

        if test_data_list:
            self.test_dataset = UCRDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            Log.info(f"Total test sequences: {len(self.test_dataset)}")
        else:
            self.test_dataset = None
            Log.error("Test dataset is empty.")

    def _load_data_files(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        file_path = os.path.join(self.data_path, self.filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        pattern = r"^\d+_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$"
        match = re.match(pattern, self.filename)
        if not match:
            raise ValueError(f"Filename {self.filename} does not match the expected pattern.")
        train_end_idx = int(match.group(2))
        anomaly_start_idx = int(match.group(3))
        anomaly_end_idx = int(match.group(4))

        data = np.loadtxt(file_path)
        data = np.nan_to_num(data)

        labels = np.zeros(len(data), dtype=int)
        labels[anomaly_start_idx - 1:anomaly_end_idx] = 1

        train_data = data[:train_end_idx]
        train_labels = labels[:train_end_idx]

        test_data = data[train_end_idx:]
        test_labels = labels[train_end_idx:]

        return [train_data], [train_labels], [test_data], [test_labels]

    def _normalize_and_filter(self, data_list: List[np.ndarray], labels_list: List[np.ndarray], fit=True) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        if not data_list:
            return [], []

        if fit:
            self.scaler = StandardScaler()
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
                self.train_dataset = UCRDataset(train_data_split, train_labels_split, seq_len=self.seq_len)
                Log.info(f"Total training sequences: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
                Log.error("Training dataset is empty.")

            if val_data_split:
                self.val_dataset = UCRDataset(val_data_split, val_labels_split, seq_len=self.seq_len)
                Log.info(f"Total validation sequences: {len(self.val_dataset)}")
            else:
                self.val_dataset = None
                Log.warning("Validation dataset is not created.")
        else:
            self.train_dataset = UCRDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            self.val_dataset = None
            Log.warning("Validation dataset is not created as val_size is set to 0.")
