import os
import torch
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from arff2pandas import *


class ECG5000_train(Dataset):
    # https://github.com/gmguarino/ecg-anomaly-detection-vae/blob/master/ecg_dataset.py
    def __init__(self, batch_size):
        with open(os.path.join('data/ECG500/ECG5000_TEST.arff')) as f:
            x_train, y_train = arff.loadarff(f)

        df_train = pd.DataFrame(x_train)
        df_train["target"] = pd.to_numeric(df_train["target"])
        df_train = df_train.astype('float32')

        # TODO Workaround to to handle last batch in LSTM hidden state
        # https://discuss.pytorch.org/t/how-to-handle-last-batch-in-lstm-hidden-state/40858

        train_size = 0
        if df_train.shape[0] % batch_size == 0:
            train_size = df_train.shape[0]
        else:
            train_size = df_train.shape[0] - (df_train.shape[0] % batch_size)

        df_train = df_train.head(train_size - 500)
        self.y_train = df_train['target']
        df_train = df_train.drop("target", axis=1)
        self.y_train = torch.tensor(self.y_train[:].values)
        self.x_train = torch.tensor(df_train[:].values)
        # https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
        super(ECG5000_train, self).__init__()

    def __len__(self):
        shape = self.x_train.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_train[index], self.y_train[index]
        return input, label


class ECG5000_val(Dataset):
    def __init__(self, batch_size):
        with open(os.path.join('data/ECG500/ECG5000_TEST.arff')) as f:
            x_train, y_train = arff.loadarff(f)

        df_train = pd.DataFrame(x_train)
        df_train["target"] = pd.to_numeric(df_train["target"])
        df_train = df_train.astype('float32')

        train_size = 0
        if df_train.shape[0] % batch_size == 0:
            train_size = df_train.shape[0]
        else:
            train_size = df_train.shape[0] - (df_train.shape[0] % batch_size)

        df_val = df_train.tail(500)
        self.y_val = df_val['target']
        df_val = df_val.drop("target", axis=1)
        self.y_val = torch.tensor(self.y_val[:].values)
        self.x_val = torch.tensor(df_val[:].values)
        super(ECG5000_val, self).__init__()

    def __len__(self):
        shape = self.x_val.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_val[index], self.y_val[index]
        return input, label


class ECG5000_test(Dataset):
    def __init__(self, batch_size):
        with open(os.path.join('data/ECG500/ECG5000_TRAIN.arff')) as f:
            x_test, y_test = arff.loadarff(f)

        df_test = pd.DataFrame(x_test)
        df_test["target"] = pd.to_numeric(df_test["target"])
        df_test = df_test.astype('float32')

        test_size = 0
        if df_test.shape[0] % batch_size == 0:
            test_size = df_test.shape[0]
        else:
            test_size = df_test.shape[0] - (df_test.shape[0] % batch_size)

        df_test = df_test.head(test_size)
        self.y_test = df_test['target']
        df_test = df_test.drop("target", axis=1)
        self.y_test = torch.tensor(self.y_test[:].values)
        self.x_test = torch.tensor(df_test[:].values)
        # https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor

        super(ECG5000_test, self).__init__()

    def __len__(self):
        shape = self.x_test.shape[0]
        return shape

    def __getitem__(self, index):
        input, label = self.x_test[index], self.y_test[index]
        return input, label


class TimeSeriesDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 1,
            num_workers: int = 16,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ECG5000_train(self.batch_size)
        self.val_dataset = ECG5000_val(self.batch_size)
        self.test_dataset = ECG5000_test(self.batch_size)

    def train_dataloader(self) -> DataLoader:
        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

        return data

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        data = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        return data

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        data = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        return data
