import os
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import CelebA
import zipfile

from arff2pandas import *
from sklearn.model_selection import train_test_split


# Add your custom dataset class here

class ECG5000(Dataset):
    # https://github.com/gmguarino/ecg-anomaly-detection-vae/blob/master/ecg_dataset.py
    """
    La classe dataset viene estesa per fornire in modo efficiente i dati ad un
    modello. Ha tre metodi principali, __init__, __len__ e __get_item__.
    """

    def __init__(self):
        with open(os.path.join('data/ECG500/ECG5000_TEST.arff')) as f:
            x_train, y_train = arff.loadarff(f)
        with open(os.path.join('data/ECG500/ECG5000_TRAIN.arff')) as f:
            x_test, y_test = arff.loadarff(f)

        df_train = pd.DataFrame(x_train)
        df_train["target"] = pd.to_numeric(df_train["target"])

        df_test = pd.DataFrame(x_test)
        df_test["target"] = pd.to_numeric(df_test["target"])

        df_train = df_train.astype('float32')
        df_test = df_test.astype('float32')

        df_train = df_train.head(4480)
        df_test = df_test.head(448)


        self.y_train = df_train['target']
        self.y_test = df_test['target']

        df_train = df_train.drop("target", axis=1)
        df_test = df_test.drop("target", axis=1)

        self.y_train = torch.tensor(self.y_train[:].values)
        self.y_test = torch.tensor(self.y_test[:].values)

        self.x_train = torch.tensor(df_train[:].values)
        self.x_test = torch.tensor(df_test[:].values)
        # https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor

        super(ECG5000, self).__init__()

    def __len__(self):
        """
        Ha la funzione di definire la quantità di dati in un dataset, così che il
        Dataloader sa quando finisce una epoca e vanno re-indicizzati e mischiati
        i dati.
        """
        return self.x_train.shape[0]

    def __getitem__(self, index):
        """
        Questa invece è la funzione che fa il lavoro pesante. Estrae i dati che
        servono e li passa al modello.
        """
        return self.x_train[index], self.y_train[index]


class TimeSeriesDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ECG5000()
        self.val_dataset = ECG5000()

    def train_dataloader(self) -> DataLoader:
        temp = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

        return temp

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        temp = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        return temp

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        temp = DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        return temp
