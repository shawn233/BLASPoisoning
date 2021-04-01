'''
Author: shawn233
Date: 2021-01-18 21:44:58
LastEditors: shawn233
LastEditTime: 2021-01-19 17:04:10
Description: PyTorch training utils
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import time
import os
import shutil
import logging
from typing import Any, Optional, Callable, Union, List

import matplotlib.pyplot as plt



def train(
        net, 
        dataset_train,
        dataset_test = None, 
        lr: float = 0.01,       # learning rate
        momentum: float = 0.9,  # for optim.SGD
        weight_decay: float = 0.01, # for SGD (0), Adam (0) and AdamW (0.01)
        gamma: float = 0.1,         # for StepLR (0.1): `lr = lr * gamma` every `decay_delay` epochs
        decay_delay: int = None,    # for StepLR (required argument): `lr` decay interval
        batch_size: int = 128,
        epochs: int = 50,
        num_workers: int = 0,   # recommend: 4 x #GPU
        optimizer: str = "adam",
        device: str = "cpu",
        display_step: int = 100,
        model_root: str = None,     # if not None, save latest and best model to model_root
        load_latest: bool = False,  # if True, continue on checkpoint (model_latest.ckpt)
        # basic plotting utility
        plot_loss: bool = False,
        plot_acc: bool = False,
        drop_last: bool = False,    # for DataLoader
        **kwargs
) -> None:
    logging.warning(f"Parameters not implemented: {kwargs}.")

    # Data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers,
                              drop_last=drop_last)
    if dataset_test is not None:
        test_loader = DataLoader(dataset_test, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    # Training device
    device = device.lower()
    if torch.cuda.is_available() and device == "cpu":
        logging.warning(f"Cuda is available but training device is CPU.")
    device = torch.device(device)
    net.to(device)

    # Optimizer
    optimizer = optimizer.lower()
    
    if optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}.")
    
    if decay_delay is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_delay,
                                            gamma=gamma)
    else:
        scheduler = None

    # Loss function
    criterion = nn.NLLLoss()

    # Plotting
    n_batch_train = np.ceil(float(len(dataset_train)) / batch_size)
    if drop_last:
        n_batch_train = n_batch_train - 1
    if plot_loss:
        train_loss_y = []
        test_loss_y = []
    if plot_acc:
        train_acc_y = []
        test_acc_y = []

    # Train by epochs
    for epoch in range(epochs):
        logging.info(f"epoch {epoch+1}/{epochs}")
        net.train()
        # statistics: running statistics are cleared every `display_step` iterations
        #             train statistics are cleared every epoch
        running_loss = 0.0
        running_correct, running_total = 0, 0
        train_total_loss = 0.0
        train_total_correct, train_total = 0, 0

        # Train
        for idx, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            # Suppose the model architecture has the last layer
            # z = self.fc_last(x)
            # then log_f = F.log_softmax(z), which applies to NLLLoss.
            log_fs = net(inputs)
            loss = criterion(log_fs, labels)
            loss.backward()
            optimizer.step()

            # statistics: loss
            running_loss += loss.item()
            train_total_loss += (loss.item() * labels.size(0))
            if plot_loss:
                train_loss_y.append(loss.item())
            
            # statistics: accuracy
            running_total += labels.size(0)
            train_total += labels.size(0)
            _, predicted = torch.max(log_fs.data, 1)
            n_correct = (predicted == labels).sum().item()
            running_correct += n_correct
            train_total_correct += n_correct
            if plot_acc:
                train_acc_y.append(n_correct / labels.size(0))
            
            # show statistics
            if idx % display_step + 1 == display_step:
                logging.info(f"batch {idx+1}\t"
                             f"[train] loss: {running_loss / display_step:.4f}\t"
                             f"acc: {100. * running_correct / running_total:4.2f}%")
                running_loss = 0.0
                running_correct, running_total = 0, 0

        train_loss = train_total_loss / train_total
        train_acc = train_total_correct / train_total

        # Validate
        if test_loader is not None:
            test_total_loss = 0.0
            test_total_correct, test_total = 0, 0
            with torch.no_grad():
                net.eval()
                for idx, data in enumerate(test_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    log_fs = net(inputs)
                    loss = criterion(log_fs, labels)
                    
                    # statistics: loss
                    test_total_loss += (loss.item() * labels.size(0))
                    
                    # statistics: accuracy
                    test_total += labels.size(0)
                    _, predicted = torch.max(log_fs.data, 1)
                    test_total_correct += (predicted == labels).sum().item()

            test_loss = test_total_loss / test_total
            test_acc = test_total_correct / test_total

            logging.info(f"==> epoch {epoch+1}/{epochs}\t"
                        f"[train] loss: {train_loss:.4f}\tacc: {100.*train_acc:4.2f}%\t"
                        f"[test] loss: {test_loss:.4f}\tacc: {100.*test_acc:4.2f}%")
        else:
            logging.info(f"==> epoch {epoch+1}/{epochs}\t"
                        f"[train] loss: {train_loss:.4f}\tacc: {100.*train_acc:4.2f}%")
        
        if scheduler is not None:
            scheduler.step()

    logging.info("Training Finished.")



from model import LeNet5
from datautils import MyMNIST
from torchvision.datasets import MNIST
from torchvision import transforms



def main():
    logging.basicConfig(level=logging.INFO)
    net = LeNet5(mnist=True)
    transform = transforms.ToTensor()
    dataset_train = MyMNIST("./dataset/mnist/", train=True, transform=transform, download=False)
    dataset_test = MyMNIST("./dataset/mnist/", train=False, transform=transform, download=False)
    params = {
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "gamma": 0.1,
        "decay_delay": None,
        "batch_size": 128,
        "epochs": 5,
        "early_stop": None,
        "num_workers": 4,
        "optimizer": "adam",
        "device": "cuda:0",
        "display_step": 100,
        "model_root": None,
        "load_latest": False,
    }
    train(net, dataset_train, dataset_test, **params)



if __name__ == "__main__":
    main()