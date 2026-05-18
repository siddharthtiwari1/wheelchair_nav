#!/usr/bin/env python3
"""
WireCNN — Cable/wire-following CNN for velocity regression.

Reconstructed from checkpoint state_dict:
  features: Conv2d(3,16,3)->BN->ReLU->MaxPool -> Conv2d(16,32,3)->BN->ReLU->MaxPool
            -> Conv2d(32,64,3)->BN->ReLU->AdaptiveAvgPool(1)
  regressor: Flatten->Dropout->Linear(64,32)->ReLU->Dropout->Linear(32,3)

Input:  (B, 3, 128, 128) RGB image
Output: (B, 3) — [forward_vel, angular_vel, lateral_vel]
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class WireCNN(pl.LightningModule):
    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),     # 0
            nn.BatchNorm2d(16),                   # 1
            nn.ReLU(inplace=True),                # 2
            nn.MaxPool2d(2),                      # 3
            nn.Conv2d(16, 32, kernel_size=3),     # 4
            nn.BatchNorm2d(32),                   # 5
            nn.ReLU(inplace=True),                # 6
            nn.MaxPool2d(2),                      # 7
            nn.Conv2d(32, 64, kernel_size=3),     # 8
            nn.BatchNorm2d(64),                   # 9
            nn.ReLU(inplace=True),                # 10
            nn.AdaptiveAvgPool2d(1),              # 11
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),                          # 0
            nn.Dropout(0.3),                       # 1
            nn.Linear(64, 32),                     # 2
            nn.ReLU(inplace=True),                 # 3
            nn.Dropout(0.3),                       # 4
            nn.Linear(32, 3),                      # 5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
