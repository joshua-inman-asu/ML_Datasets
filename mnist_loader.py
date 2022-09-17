#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:21:25 2022

@author: apple
"""

import torchvision
from scipy import stats
import numpy as np
import torch
import random
from torch.utils.data import SubsetRandomSampler #Dataset, DataLoader,


# Gets the data for binary MNIST
def get_data_full(batch_size_train=32, batch_size_test=32, noisy_prob=.3, imb_a = int(5000), imb_b = int(5000)):
    # Gets the training set and downloads it to './data' if not already downloaded
    train_set = torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor()
    ]))

    # Section removes all data from the set that is not 1s or 7s
    class_dic = {}
    for idx, target in enumerate(train_set.targets.tolist()):
        if target not in class_dic.keys():
            class_dic[target] = [idx]
        else:
            class_dic[target].append(idx)
    num_classes = 10
    imbalance = [0,imb_a,0,0,0,0,0,imb_b,0,0]
    new_train_set_id = {}
    new_train_set = []
    for idx, im in enumerate(imbalance):
        sample = random.sample(class_dic[idx],imbalance[idx])
        new_train_set_id[idx] = sample
        for ids in new_train_set_id[idx]:
            new_train_set.append(ids)
            if train_set.targets[ids] == 1:
                train_set.targets[ids] = 0
            elif train_set.targets[ids] == 7:
                train_set.targets[ids] = 1

    # Section adds label noise to the data
    noisy_indices = new_train_set.copy()
    random.shuffle(noisy_indices)
    noisy_indices = noisy_indices[:int(noisy_prob*len(noisy_indices))]

    for idx in noisy_indices:
        label = train_set.targets.tolist()[idx]
        choices = list(range(2))
        choices.remove(label)
        new_label = np.random.choice(choices)
        train_set.targets[idx] = int(new_label) #torch.LongTensor([new_label])

    dsamples = np.array([len(new_train_set_id[i]) for i in range(10)])
    # Final training data
    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,sampler=SubsetRandomSampler(new_train_set),drop_last=True)

    # Loads the test set and downloads it to './data'
    test_set = torchvision.datasets.MNIST('./data', train=False, download=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                                ]))

    # Makes the test set binary 1s and 7s
    new_test_set = []
    for idx, target in enumerate(test_set.targets.tolist()):
        if target == 1:
            test_set.targets[idx] = 0
            new_test_set.append(idx)
        elif target == 7:
            test_set.targets[idx] = 1
            new_test_set.append(idx)

    # Final test set
    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test,sampler=SubsetRandomSampler(new_test_set))
    return train_subset_loader, test_subset_loader