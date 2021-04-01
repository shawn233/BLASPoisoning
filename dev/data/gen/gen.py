'''
Author: shawn233
Date: 2021-03-14 19:11:50
LastEditors: shawn233
LastEditTime: 2021-03-14 20:50:03
Description: Generate data for 2-layer network
'''

import os
import numpy as np


def softmax(Z):
    exp_Z = np.exp(Z)
    rowsum = np.sum(exp_Z, axis=1)
    Y = np.zeros_like(exp_Z)

    for i in range(len(rowsum)):
        for j in range(len(exp_Z[0])):
            Y[i][j] = exp_Z[i][j] / rowsum[i]

    return Y


def main():
    n_features = 5
    n_labels = 2

    n_samples = 100

    np.random.seed(15)

    W = np.random.random((n_features, n_labels))
    X = np.random.random((n_samples, n_features))
    B = np.random.random((n_labels))
    Z = np.matmul(X, W) + B
    # print(W)
    # print(B)
    # print(X)
    # print(Z)
    Y = softmax(Z)
    labels = np.argmax(Y, axis=1)

    label_cnt = []
    for i in range(n_labels):
        label_cnt.append(np.sum(labels==i))

    print(label_cnt)

    X_write = ["\t".join(list(X[i].astype(str)))+"\n" for i in range(X.shape[0])]
    labels_write = "\n".join(list(labels.astype(str)))

    with open("gen-features.txt", "w") as f:
        f.writelines(X_write)

    with open("gen-labels.txt", "w") as f:
        f.write(labels_write)

    # search for a good seed
    # stop = False
    # seed = 0
    # target_min = 40
    # while not stop:
    #     np.random.seed(seed)

    #     W = np.random.random((n_features, n_labels))
    #     X = np.random.random((n_samples, n_features))
    #     B = np.random.random((n_labels))
    #     Z = np.matmul(X, W) + B
    #     # print(W)
    #     # print(X)
    #     # print(Z)
    #     Y = softmax(Z)
    #     labels = np.argmax(Y, axis=1)

    #     label_cnt = []
    #     for i in range(n_labels):
    #         label_cnt.append(np.sum(labels==i))

    #     if seed % 1000 == 0:
    #         print(seed)

    #     if min(label_cnt) > target_min:
    #         stop = True
    #         print(seed)
    #         print(label_cnt)
    #     else:
    #         seed += 1


if __name__ == "__main__":
    main()
