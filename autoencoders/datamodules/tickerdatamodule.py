import os
from math import floor
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co


class TickerDataset(Dataset):

    def __init__(self, data_file: str, time_period: int, preprocessing: bool):
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file, index_col=[0])
            if 'sector' in self.data.columns:
                self.data.drop(columns=['sector'], axis=1, inplace=True)
            if preprocessing:
                self.preprocessing()
            self.prepare_data(time_period)
        else:
            raise FileNotFoundError

    def preprocessing(self) -> None:
        self.data = self.data.dropna(axis=1).pct_change().fillna(0).T

    def prepare_data(self, time_period: int) -> None:
        if time_period != 0 or -1:
            self.data = np.vstack(
                [self.data.values[:, time_period * i: time_period * (i + 1)]
                 for i in range(floor(self.data.shape[1] / time_period))])

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.data[index]).float().unsqueeze(0)

    def __len__(self) -> int:
        return self.data.shape[0]


class TickerDataModule(LightningDataModule):

    def __init__(self, data_path: str,
                 batch_size: int = 16,
                 time_period: int = 30,
                 preprocessing: bool = True):
        super(TickerDataModule, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.time_period = time_period
        self.preprocessing = preprocessing
        self.test = None
        self.train = None
        self.val = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_val, self.test = train_test_split(
            TickerDataset(self.data_path, self.time_period, self.preprocessing), train_size=0.8)
        self.train, self.val = train_test_split(train_val, train_size=0.85)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

