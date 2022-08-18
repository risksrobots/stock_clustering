from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT, EPOCH_OUTPUT


class MLPAutoEncoder(LightningModule):

    def __init__(self, in_features: int, latent_features: int):
        super(MLPAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),

            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),

            nn.Linear(in_features, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
            nn.Dropout(0.3),

            nn.Linear(latent_features, latent_features),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, latent_features),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(latent_features, latent_features),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(latent_features, in_features),
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),

            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(in_features, in_features)
        )

        self.time_start = time.time()

    def predict_step(self, batch: Any) -> Any:
        batch = batch.squeeze(1)
        return self.encoder(batch)

    def forward(self, x: Tensor, *args, **kwargs) -> Any:
        x = x.squeeze(1)
        latent = self.encoder(x)
        return self.decoder(latent)

    def training_step(self, batch: Tensor, *args, **kwargs) -> STEP_OUTPUT:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('train_time', time.time() - self.time_start, prog_bar=True)

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        def adjust_lr(epoch):
            if epoch < 100:
                return 0.003
            if 100 <= epoch < 120:
                return 0.0003
            if 120 <= epoch < 150:
                return 0.000003
            if 150 <= epoch < 200:
                return 0.0000003
            else:
                return 0.00000003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]
