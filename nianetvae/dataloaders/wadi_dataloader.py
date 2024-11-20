import os
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from log import Log
from nianetvae.dataloaders import BaseDataLoader


class WADIDataset(Dataset):
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


class WADIDataLoader(BaseDataLoader):
    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list, train_labels_list, test_data_list, test_labels_list = self._load_data_files()

        train_data_list, train_labels_list = self._normalize_and_filter(train_data_list, train_labels_list, fit=True)
        test_data_list, test_labels_list = self._normalize_and_filter(test_data_list, test_labels_list, fit=False)

        self._split_train_validation(train_data_list, train_labels_list)

        if test_data_list:
            self.test_dataset = WADIDataset(test_data_list, test_labels_list, seq_len=self.seq_len)
            Log.info(f"Total test sequences: {len(self.test_dataset)}")
        else:
            self.test_dataset = None
            Log.error("Test dataset is empty.")

    def _load_data_files(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        normal_path = os.path.join(self.data_path, "WADI_14days.csv")
        attack_path = os.path.join(self.data_path, "WADI_attackdata.csv")

        # Load normal data
        normal = pd.read_csv(normal_path, sep=',', skiprows=[0, 1, 2, 3], skip_blank_lines=True)
        Log.info(f"Number of rows in training data: {len(normal)}")
        normal = normal.drop(normal.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)
        normal = normal.astype(float).fillna(0)
        down_rate = 5
        normal = normal.groupby(np.arange(len(normal)) // down_rate).mean().values

        # Load attack data and labels
        attack = pd.read_csv(attack_path, sep=",")
        Log.info(f"Number of rows in testing data: {len(attack)}")
        labels = self._generate_attack_labels(attack)
        attack = attack.drop(attack.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)
        attack = attack.astype(float).fillna(0)
        attack = attack.groupby(np.arange(len(attack)) // down_rate).mean().values

        # Downsample labels
        labels_down = self._downsample_labels(labels, down_rate)

        return [normal], [np.zeros(len(normal))], [attack], [np.array(labels_down)]

    @staticmethod
    def _generate_attack_labels(df: pd.DataFrame) -> List[float]:
        labels = []
        for index, row in df.iterrows():
            date_temp = datetime.strptime(row['Date'], "%m/%d/%Y")
            time_temp = datetime.strptime(row['Time'], "%I:%M:%S.%f %p")

            # October 9, 2017
            if date_temp == datetime.strptime('10/9/2017', '%m/%d/%Y'):
                if datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('7:50:16.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue

            # October 10, 2017
            if date_temp == datetime.strptime('10/10/2017', '%m/%d/%Y'):
                if datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('10:34:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('11:24:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('11:44:50.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('1:50:40.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('2:59:55.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('5:49:40.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('10:56:27.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue

            # October 11, 2017
            if date_temp == datetime.strptime('10/11/2017', '%m/%d/%Y'):
                if datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('11:31:20.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('11:47:00.000 AM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('12:05:00.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('12:10:52.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('12:25:36.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue
                elif datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') <= time_temp <= datetime.strptime('3:37:00.000 PM', '%I:%M:%S.%f %p'):
                    labels.append(1.0)
                    continue

            # Default to 'Normal' if no conditions are met
            labels.append(0.0)

        return labels


    @staticmethod
    def _downsample_labels(labels: List[float], down_rate: int) -> List[float]:
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
        return labels_down

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
                self.train_dataset = WADIDataset(train_data_split, train_labels_split, seq_len=self.seq_len)
                Log.info(f"Total training sequences: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
                Log.error("Training dataset is empty.")

            if val_data_split:
                self.val_dataset = WADIDataset(val_data_split, val_labels_split, seq_len=self.seq_len)
                Log.info(f"Total validation sequences: {len(self.val_dataset)}")
            else:
                self.val_dataset = None
                Log.warning("Validation dataset is not created.")
        else:
            self.train_dataset = WADIDataset(train_data_list, train_labels_list, seq_len=self.seq_len)
            self.val_dataset = None
            Log.warning("Validation dataset is not created as val_size is set to 0.")
