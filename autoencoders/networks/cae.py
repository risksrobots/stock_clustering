__all__ = ["Conv1dAutoEncoder"]

import numpy as np
import pytorch_lightning as pl
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def init_weights(m):
    """
    Simple weight initialization
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.fill_(0.01)


class Conv1dAutoEncoder(pl.LightningModule):
    """
    Main block of convolutional event clustering
    encoder-decoder architecture allows to create representation of Cohortney features
    """

    def __init__(
        self,
        in_channels: int,
        n_latent_features: int,
        seq_len: int = 100
    ):
        super().__init__()
        print(in_channels)
        self.out = n_latent_features
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=3),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3),
        )
        self.encoder.apply(init_weights)

        self.bottle_neck_encoder = nn.Linear(in_features=seq_len - 12, out_features=self.out)

        self.bottle_neck_decoder = nn.Linear(in_features=self.out, out_features=seq_len - 12)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(in_channels=512, out_channels=in_channels, kernel_size=3),
        )
        self.decoder.apply(init_weights)

        self.train_index = 0
        self.val_index = 0
        self.final_labels = None
        self.time_start = time.time()

    def forward(self, x):
        """
        Returns embeddings
        """
        latent = self.encoder(x)
        latent = self.bottle_neck_encoder(latent)
        return latent

    def predict_step(self, x):
        return self(x)

    def training_step(self, batch, batch_idx):
        x = batch
        latent = self(x)
        # print('Latent shape is' + str(latent.shape))
        loss = torch.nn.MSELoss()(self.decoder(self.bottle_neck_decoder(latent)), x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_time", time.time() - self.time_start, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(self.bottle_neck_decoder(latent)), x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx: int):
        x = batch
        latent = self(x)
        loss = torch.nn.MSELoss()(self.decoder(self.bottle_neck_decoder(latent)), x)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "latent": latent}

    def test_epoch_end(self, outputs):
        pass

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
