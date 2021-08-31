'''
Author: shawn233
Date: 2021-04-02 15:13:26
LastEditors: shawn233
LastEditTime: 2021-04-02 16:04:08
Description: Export PyTorch model
'''

import os
import torch
import numpy as np
import pandas as pd


def main():
    ckpt = torch.load("./iris/best.ckpt")
    print(f'epoch: {ckpt["epoch"]}, training loss: {ckpt["train_loss"]:.4f}, '
            f'training acc: {100.*ckpt["train_acc"]:.2f}%')
    model = ckpt["model_state_dict"]
    
    with open("./iris/best.txt", "w") as f:
        for name in model:
            arr = model[name].numpy()
            var_name = name.split(".")[1].upper()
            s = [var_name+":"]
            
            if var_name == "WEIGHT":
                for i in range(len(arr)):
                    s.append("\t".join([str(w) for w in arr[i]]))
            elif var_name == "BIAS":
                s.append("\t".join([str(b) for b in arr]))
            else:
                raise ValueError(f"Unknown var_name {var_name}")

            # print (s)
            f.write("\n".join(s)+"\n")

    print("Done!")


if __name__ == "__main__":
    main()