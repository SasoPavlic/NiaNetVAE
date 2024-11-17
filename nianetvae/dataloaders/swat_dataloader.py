import os
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class SWATDataset(Dataset):
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
            self.sequences = torch.empty((0, self.seq_len, data_list[0].shape[1]))
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
        return {'signal': signal, 'target': target, 'ts_id': ts_id}


class SWATDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list, train_labels_list, test_data_list, test_labels_list = self._load_data_files()

        train_data_list, train_labels_list = self._normalize_and_filter(train_data_list, train_labels_list, fit=True)
        test_data_list, test_labels_list = self._normalize_and_filter(test_data_list, test_labels_list, fit=False)

        self._split_train_validation(train_data_list, train_labels_list)

        if test_data_list:
            self.test_dataset = SWATDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            Log.info(f"Total test sequences: {len(self.test_dataset)}")
        else:
            self.test_dataset = None
            Log.error("Test dataset is empty.")

    def _load_data_files(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        normal_path = os.path.join(self.data_path, "SWaT_Dataset_Normal_v1.csv")
        attack_path = os.path.join(self.data_path, "SWaT_Dataset_Attack_v0.csv")

        # Load normal data
        normal = pd.read_csv(normal_path).drop(["Timestamp", "Normal/Attack"], axis=1)
        print(f"Number of rows in training data: {len(normal)}")
        for col in normal.columns:
            normal[col] = normal[col].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype(float)
        down_rate = 5
        normal = normal.groupby(np.arange(len(normal)) // down_rate).mean().values

        # Load attack data and labels
        attack = pd.read_csv(attack_path, sep=";")
        print(f"Number of rows in testing data: {len(attack)}")
        labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
        attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
        for col in attack.columns:
            attack[col] = attack[col].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)
        attack = attack.groupby(np.arange(len(attack)) // down_rate).mean().values

        # Downsample labels
        labels_down = []
        for i in range(len(labels) // down_rate):
            if labels[down_rate * i:down_rate * (i + 1)].count(1.0):
                labels_down.append(1.0)
            else:
                labels_down.append(0.0)
        if labels[down_rate * (i + 1):].count(1.0):
            labels_down.append(1.0)
        else:
            labels_down.append(0.0)

        return [normal], [np.zeros(len(normal))], [attack], [np.array(labels_down)]

    def _normalize_and_filter(self, data_list: List[np.ndarray], labels_list: List[np.ndarray], fit=True) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        if not data_list:
            return [], []

        if fit:
            self.scaler = MinMaxScaler()
            all_data = np.concatenate(data_list).reshape(-1, data_list[0].shape[1])
            self.scaler.fit(all_data)
        else:
            if not hasattr(self, 'scaler'):
                raise AttributeError("Scaler not initialized. Ensure training data is normalized before test data.")

        data_list = [self.scaler.transform(data) for data in data_list]

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
                self.train_dataset = SWATDataset(train_data_split, train_labels_split, seq_len=self.seq_len)
                Log.info(f"Total training sequences: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
                Log.error("Training dataset is empty.")

            if val_data_split:
                self.val_dataset = SWATDataset(val_data_split, val_labels_split, seq_len=self.seq_len)
                Log.info(f"Total validation sequences: {len(self.val_dataset)}")
            else:
                self.val_dataset = None
                Log.warning("Validation dataset is not created.")
        else:
            self.train_dataset = SWATDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            self.val_dataset = None
            Log.warning("Validation dataset is not created as val_size is set to 0.")
