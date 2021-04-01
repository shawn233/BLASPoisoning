'''
Author: shawn233
Date: 2021-04-01 03:53:14
LastEditors: shawn233
LastEditTime: 2021-04-01 20:20:43
Description: Models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



class TwoLayerFC(nn.Module):
    
    def __init__(self, *args):
        super(TwoLayerFC, self).__init__() # necessary otherwise error
        self.fc = nn.Linear(in_features=4, out_features=3, bias=True)    

    def forward(self, x):
        z = self.fc(x)
        log_fs = F.log_softmax(z, dim=1)
        
        return log_fs



def main():
    pass


if __name__ == "__main__":
    main()