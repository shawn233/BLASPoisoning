'''
Author: shawn233
Date: 2021-04-01 03:48:28
LastEditors: shawn233
LastEditTime: 2021-04-10 11:03:56
Description: Train model
'''

import os
import argparse
import logging
from typing import Any, Optional, Callable, Union, List
import numpy as np
import torch
# from torchvision import transforms
from torch.utils.data import DataLoader

from model import TwoLayerFC
from trainutils import train, test
from datautils import SmallDataset


class Iris(SmallDataset):

    def __init__(
            self,
            root: str,
            train: bool = True,
            train_ratio: float = 0.8,
    ) -> None:
        super(Iris, self).__init__(root, train)

        features = []
        with open(os.path.join(self.root, "iris-features.txt")) as f:
            for line in f:
                line = line.strip()
                features.append([float(i) for i in line.split('\t')])
        
        _data = torch.tensor(features)
        print(_data.dtype, _data.shape)

        labels_ = []
        with open(os.path.join(self.root, "iris-labels.txt")) as f:
            for line in f:
                line = line.strip()
                labels_.append(int(line))

        _labels = torch.tensor(labels_)
        print(_labels.dtype, _labels.shape)

        train_ind = int(train_ratio * len(_labels))
        if train:
            self.data = _data[:train_ind]
            self.labels = _labels[:train_ind]
        else:
            self.data = _data[train_ind:]
            self.labels = _labels[train_ind:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    net = TwoLayerFC()
    # transform = transforms.ToTensor()
    if args.train:
        iris = Iris(root="../data/iris/", train=True)
        iris_test = Iris(root="../data/iris/", train=False)
        params = {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0,
            "gamma": 0.1,
            "decay_delay": None,
            "batch_size": 16,
            "epochs": 200,
            "early_stop": None,
            "num_workers": 4,
            "optimizer": "adam",
            "device": "cpu",
            "display_step": 1,
            "model_root": "./iris",
            "load_latest": False,
        }
        train(net, iris, iris_test, **params)
    else:
        iris_test = Iris(root="../data/iris/", train=False)
        params = {
            "ckpt_path": "./iris/best.ckpt",
            "batch_size": 64,
            "device": "cpu",
        }
        test(net, iris_test, **params)



if __name__ == "__main__":
    main()
