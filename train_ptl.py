#!/usr/bin/env python3

import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data.dataset as ds
from torch.utils.data import DataLoader

from sklearn.datasets import load_svmlight_file
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from test_tube import HyperOptArgumentParser


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
    def __init__(self, layer_size: int, nb_layers: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(103, layer_size),
            nn.ReLU(),
            # *[nn.Linear(layer_size, layer_size) for i in range(nb_layers-2)],
            nn.Linear(layer_size, 14)
        )
        self.train_prc = pl.metrics.Precision(num_classes=14, multilabel=True)
        self.val_prc = pl.metrics.Precision(num_classes=14, multilabel=True)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)

        self.train_prc(y_hat, y)
        self.log('train_prc', self.train_prc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('val_loss', loss)

        self.val_prc(y_hat, y)
        self.log('val_prc', self.val_prc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train_main(args):
    train = LibSVMDataset(args.train)
    val = LibSVMDataset(args.val)

    model = YestMultiClassifier(args.layer_size, args.nb_layers)

    # log = pl_loggers.TestTubeLogger('tt_logs', name="nb_layers")
    log = pl_loggers.TensorBoardLogger('lightning_logs')
    log.log_hyperparams(args)

    trainer = pl.Trainer(logger=log, callbacks=[EarlyStopping(monitor='val_prc')]) #fast_dev_run=True,
    trainer.fit(model,
        DataLoader(train, batch_size=10, shuffle=True),
        DataLoader(val, batch_size=10)
    )
    #torch.save(model.state_dict(), args.model)

def main():
    parser = HyperOptArgumentParser(
        description='Train a PyTorch Lightning model on Yest dataset',
        strategy='random_search'
    )

    parser.opt_list('--nb_layers', default=2, type=int, tunable=False, options=[2, 4, 8])
    parser.opt_range('--layer_size', default=20, type=int, tunable=False, low=10, high=200, nb_samples=10, help="size of the hidden layer")

    parser.add_argument('--model', default="model.ptl", help="path to save the model")
    parser.add_argument('--train', default="yeast_train.svm", help="path to the training data")
    parser.add_argument('--val', default="yeast_test.svm", help="path to the training data")

    hparams = parser.parse_args()
    hparams.optimize_parallel_cpu(train_main, nb_trials=20, nb_workers=8)


if __name__ == "__main__":
    main()