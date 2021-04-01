'''
Author: shawn233
Date: 2021-04-01 03:48:28
LastEditors: shawn233
LastEditTime: 2021-04-01 04:24:52
Description: Train model
'''

import os
import logging
import numpy as np
from torchvision import transforms

from model import TwoLayerFC
from trainutils import train
from datautils import SmallDataset


class Iris(SmallDataset):

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Iris, self).__init__(root, train, transform, target_transform, download)

        features = []
        with open(os.path.join(self.root, "iris-features.txt")) as f:
            for line in f:
                line = line.strip()
                features.append([float(i) for i in line.split('\t')])
        
        self.data = np.asarray(features)
        print(self.data.shape)

        labels_ = []
        with open(os.path.join(self.root, "iris-labels.txt")) as f:
            for line in f:
                line = line.strip()
                labels_.append(int(line))

        self.labels = np.asarray(labels_)
        print(self.labels.shape)




def main():
    logging.basicConfig(level=logging.INFO)
    net = TwoLayerFC()
    transform = transforms.ToTensor()
    iris = Iris(root="../data/iris/", train=True)
    params = {
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "gamma": 0.1,
        "decay_delay": None,
        "batch_size": 16,
        "epochs": 10,
        "early_stop": None,
        "num_workers": 4,
        "optimizer": "adam",
        "device": "cuda:0",
        "display_step": 10,
        "model_root": None,
        "load_latest": False,
    }
    train(net, iris, **params)



if __name__ == "__main__":
    main()