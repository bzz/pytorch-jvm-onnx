#!/usr/bin/env python3

import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data.dataset as ds
from torch.utils.data import DataLoader

from sklearn.datasets import load_svmlight_file
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class LibSVMDataset(ds.Dataset):
    def __init__(self, libsvm_file):
        self.X, self.y = load_svmlight_file(libsvm_file, multilabel=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X.getrow(idx).toarray()
        x = torch.tensor(x[0]).float()

        y_labels = torch.tensor(self.y[idx]).long()
        y = F.one_hot(y_labels, num_classes=14).sum(dim=0).float()
        return x, y

class YestMultiClassifier(pl.LightningModule):
    def __init__(self, n: int):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(103, n), nn.ReLU(), nn.Linear(n, 14))
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('val_loss', loss)

        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    parser = argparse.ArgumentParser(
        description='Train a PyTorch Lightning model on Yest dataset',
    )

    parser.add_argument('--model', default="model.ptl", help="path to save the model")
    parser.add_argument('--train', default="yeast_train.svm", help="path to the training data")
    parser.add_argument('--val', default="yeast_test.svm", help="path to the training data")

    parser.add_argument('--hidden', type=int, default=100, help="size of the hidden layer")
    args = parser.parse_args()

    train = LibSVMDataset(args.train)
    val = LibSVMDataset(args.val)

    model = YestMultiClassifier(args.hidden)
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='val_acc')]) #fast_dev_run=True,
    trainer.fit(model,
        DataLoader(train, batch_size=10, shuffle=True),
        DataLoader(val, batch_size=10)
    )

    torch.save(model.state_dict(), args.model)


if __name__ == "__main__":
    main()