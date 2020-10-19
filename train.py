#!/usr/bin/env python3

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as ds
from torch.utils.data import DataLoader
from sklearn.datasets import load_svmlight_file

SEED = 2334
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(103, 103)
        self.fc2 = nn.Linear(103, 14)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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


def main():
    parser = argparse.ArgumentParser(   
        description='Train a multi-class classification PyTorch model on Yeast gene dataset',
    )

    parser.add_argument('--model', help="path to save the model stapshot")
    parser.add_argument('--train_data', default="yeast_train.svm", help="path to the training data")
    parser.add_argument('--val_data', default="yeast_test.svm", help="path to the training data")
    args = parser.parse_args()

    if not args.model:
        test_data = LibSVMDataset(args.val_data)
        debug_print(1, test_data)
        return

    # hyperparams
    num_epochs = 15
    learning_rate = 1e-3

    model = NNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoader(LibSVMDataset(args.train_data), batch_size=10, shuffle=True)
    val_dataloader = DataLoader(LibSVMDataset(args.val_data), batch_size=10)
    for epoch in range(num_epochs):

        # training loop
        train_losses = []
        for batch_idx, batch in enumerate(train_dataloader, 1):
            x, y = batch

            y_hat = model(x)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        # validation loop
        with torch.no_grad():
            val_losses = []
            for val_batch in val_dataloader:
                x, y = val_batch

                y_hat = model(x)
                val_losses.append(criterion(y_hat, y).item())
            val_loss = np.mean(val_losses)
            train_loss = np.mean(train_losses)
            print(f"Epoch: {epoch}, train_loss: {train_loss:5.3}, val_loss: {val_loss:5.3}")

    torch.save(model.state_dict(), args.model)



def debug_print(limit, dataset):
        print(f"Test Dataset ({len(dataset)} total)")
        for index, data in zip(range(limit), dataset):
            x, y = data
            print(f"x {x.size()}, {x.dtype}")
            print(f"y {y.size()}: {y}\n")

        print(f"Test DataLoader (no batch)")
        test_dl = DataLoader(dataset, batch_size=None)
        for index, data in zip(range(limit), test_dl):
            x, y = data
            print(f"x {x.size()}")
            print(f"y {y.size()}\n")
           

        print(f"Test DataLoader (batch=1)")
        test_dl = DataLoader(dataset, batch_size=1)
        for index, data in zip(range(limit), test_dl):
            x, y = data
            print(f"x {x.size()}")
            print(f"y {y.size()}\n")


if __name__ == "__main__":
    main()