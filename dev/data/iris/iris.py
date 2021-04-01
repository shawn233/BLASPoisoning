'''
Author: shawn233
Date: 2021-03-06 19:20:17
LastEditors: shawn233
LastEditTime: 2021-03-14 15:38:56
Description: Processing Iris dataset
'''

import os
from numpy.random import shuffle
import numpy as np


def main():
    label_dict = {
        "Iris-setosa": "0",
        "Iris-versicolor": "1",
        "Iris-virginica": "2",
    }

    data = []
    with open("./iris.data", "r") as iris:
        for line in iris:
            line = line.strip().split(',')
            if len(line) == 1:
                continue
            line[-1] = label_dict[line[-1]] # converts label to integer
            data.append(line)
    
    shuffle(data)

    features_arr = []
    for d in data:
        features_arr.append([float(entry) for entry in d[:-1]])
    features_arr = np.asarray(features_arr)
    # print(features_arr)
    # feature normalization
    min_val = np.min(features_arr, axis=0)
    max_val = np.max(features_arr, axis=0)
    features_arr = ( features_arr - min_val ) / ( max_val - min_val )
    
    features = ["\t".join(list(features_arr[i].astype(str))) + "\n" for i in range(features_arr.shape[0])]
    # print(features[100])
    # print(len(features))

    labels = [d[-1]+'\n' for d in data]
    # print(labels)
    # print(len(labels))

    with open("iris-features.txt", "w") as f:
        f.writelines(features)

    with open("iris-labels.txt", "w") as f:
        f.writelines(labels)

    print("Done!")


if __name__ == "__main__":
    main()