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


def init_weights(m):
    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
    m.bias.data.fill_(0.01)


class LSTMAutoEncoder(LightningModule):

    def __init__(self, seq_len, n_features, embedding_dim=16, num_layers: int = 10):
        super(LSTMAutoEncoder, self).__init__()

        self.encoder1 = nn.LSTM(input_size=n_features,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)
        self.encoder2 = nn.LSTM(input_size=embedding_dim * 2,
                                hidden_size=embedding_dim,
                                num_layers=1,
                                batch_first=True)

        self.decoder1 = nn.LSTM(input_size=embedding_dim,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)
        self.decoder2 = nn.LSTM(input_size=embedding_dim * 2,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True)

        self.output_layer = nn.Linear(embedding_dim * 2, n_features)

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.time_start = time.time()

    def predict_step(self, x: Any) -> Any:
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        x, (hidden_n, _) = self.encoder1(x)
        x, (hidden_n, _) = self.encoder2(x)

        return hidden_n[-1]

    def forward(self, x: Tensor, *args, **kwargs) -> Any:
        batch_size = x.shape[0]
        # print(x.shape)
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(x.shape)
        x, (hidden_n, _) = self.encoder1(x)
        x, (hidden_n, _) = self.encoder2(x)

        h = hidden_n[-1].unsqueeze(0)
        h = h.repeat(self.seq_len, 1, 1)
        h = torch.permute(h, (1, 0, 2))
        h, _ = self.decoder1(h)
        h, _ = self.decoder2(h)

        return self.output_layer(h)

    def training_step(self, batch: Tensor, *args, **kwargs) -> STEP_OUTPUT:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('train_time', time.time() - self.time_start, prog_bar=True)

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch_rec = self(batch)
        # print(batch.shape)
        # print(batch_rec.shape)
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
            if epoch < 20:
                return 0.003
            if 20 <= epoch < 50:
                return 0.001
            if 50 <= epoch < 80:
                return 0.0003
            if 80 <= epoch < 120:
                return 0.00003
            else:
                return 0.000003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]