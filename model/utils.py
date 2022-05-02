#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import sys

def Initialize_Model(args):
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        # TrainSet_per_client[0][0]--client 0, first sample =>(<PIL image>, label)
        # img_size = TrainSet_per_client[0][0][0].shape
        img_size = (1, 28, 28)
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        print('Error: unrecognized model')
        sys.exit(1)
    return global_model



def dir_sampling(dataset, num_clients, alpha=0.5):
    """
    dirichlet_distribution_sampling
    input:  dataset, num_clients, alpha
    return: dirichlet distributed samples stored in a dictionary,
            keys = client id, values = sample index
    param:  dataset--training dataset, i.e.,
            data = datasets.FashionMNIST(root='data/FMNIST',
                                    download=True,train=True)
            alpha-- controls the dirichlet distribution, is a list of the same
                    length as #classes, could be equal or unequal,
                    i.e., alpha = [1,1,1,1] or [1,1,100,1] for class=4,
                    the later will heavily concentrate distribution on class 3
    """
    min_size = 0
    # if dataset = ConcatDataset(MNIST_Train, MNIST_Test)
    # then num_classes = len(dataset.datasets[0].classes)
    num_classes = len(dataset.classes)
    num_all_data = len(dataset)  # dataset.shape = 60000,28,28 for FMNIST
    client_data_idx_map = {}
    least_samples = 10
    while min_size < least_samples:
        data_index = [[] for _ in range(num_clients)]
        # data_index is a list stores data_index (which is also a list)
        # for clients. initialized as empty and increases step by step
        # equivalent to a sub-dataset
        for k in range(num_classes):
            idx_of_class_k = np.where(dataset.targets == k)[0]
            # locate index belong to data from class k
            np.random.shuffle(idx_of_class_k)
            # introduce randomness?
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # generate sampling probabilities, [p_1,p_2,...p_K] \sum_{i=1}^{K}(p_i)=1
            proportions = np.array([p * (len(idx_j) < num_all_data / num_clients)
                                    for p, idx_j in zip(proportions, data_index)])
            # check the probabilities list, if client j has gain enough samples,
            # then his sampling probability is set to 0 for this round and afterward
            proportions = proportions / proportions.sum()
            # resize the prob list
            proportions = (np.cumsum(proportions) * len(idx_of_class_k)).astype(int)[:-1]
            # calculate the cumulative sum of proba list, rule out the last element(which
            # i think is unnecessary)
            data_index = [idx_j + idx.tolist() for idx_j, idx in
                          zip(data_index, np.split(idx_of_class_k, proportions))]
            # update data_index, distribute samples from class k accorss all clients
        min_size = min([len(idx_j) for idx_j in data_index])
        # calulate min_size to see whether need to re-sampling

    for j in range(num_clients):
        np.random.shuffle(data_index[j])
        client_data_idx_map[j] = data_index[j]

    return client_data_idx_map


def iid_sampling(dataset, num_clients):
    """
    IID sampling
    input: dataset, num_clients
    return: IID samples stored in a dictionary, keys = client id, values = sample index
    param:
    dataset--training dataset
    """
    num_items = int(len(dataset) / num_clients)
    # dividing dataset into num_clients parts equally
    client_data_idx_map, all_index = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        # sampling num_items items from dataset, with no repeat
        selected_idx = np.random.choice(all_index, int(num_items), replace=False)
        client_data_idx_map[i] = selected_idx
        # a single sampling will not repeat, however second sampling will
        # coincide with the first at probability, so should update the
        # dataset after each sampling
        all_index = list(set(all_index) - set(selected_idx))
    return client_data_idx_map


# fig = plot_dis(Train_set_per_client)

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    """
    Apr 29 
    Now returns two list of list. "Train_set_per_client" and "Test_set_per_client"
    Train_set_per_client[i] is the training set of client i
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)
        # test_set = datasets.CIFAR10(data_dir, train=False, download=True,
        #                             transform=apply_transform)
    elif args.dataset == 'mnist':
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        # test_set = datasets.MNIST(data_dir, train=False, download=True,
        #                           transform=apply_transform)
    elif args.dataset == 'fmnist':
        data_dir = './data/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)
        # test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
        #                               transform=apply_transform)
    if args.iid:
        # client_data_idx_map is a dictionary
        # keys=client index, values = list of sample index
        client_data_idx_map = iid_sampling(train_set, args.num_clients)
    else:
        client_data_idx_map = dir_sampling(train_set, args.num_clients, args.alpha)

    # Train_set_per_client & Test_set_per_client is a dictionary
    # keys=client index, values = sub-dataset
    Train_set_per_client, Test_set_per_client = {}, {}
    for idx in range(args.num_clients):
        # random.shuffle(client_data_idx_map[idx])
        length = len(client_data_idx_map[idx])
        Train_set_per_client[idx] = torch.utils.data.Subset(
            train_set, client_data_idx_map[idx][:round(0.8 * length)])
        Test_set_per_client[idx] = torch.utils.data.Subset(
            train_set, client_data_idx_map[idx][round(0.8 * length):])

    return Train_set_per_client, Test_set_per_client


def plot_dis(dict):
    """
    input:  a dictionary {"client id": subset}
    output: a heatmap figure plot by seaborn
    """
    num_clients = len(dict.keys())
    num_classes = len(dict[0].dataset.classes)
    count_matrix = [[] for _ in range(len(dict.keys()))]
    for client_id in dict.keys():
        label = []
        # this is not working rigth. Access a subset 1. directly through index, 0,1,...,len(subset)
        # or 2. subset.dataset[indices]
        # for index in dict[client_id].indices:
        for index in range(len(dict[client_id])):
            # dict[client_id][index] is a tuple, (<PIL.image>, label)
            label.append(dict[client_id][index][1])
        count_matrix[client_id] = [label.count(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 10))
    figure = sns.heatmap(count_matrix, annot=True, annot_kws={'size': 10},
                         fmt='.20g', cmap='Greens', ax=ax)
    figure.set(xlabel='class index', ylabel='client index')
    return figure


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Total #users  :      \t{args.num_clients}')
    print(f'    Fraction of users  : \t{args.frac}')
    print(f'    Local Batch size   : \t{args.local_bs}')
    print(f'    Local Epochs       : \t{args.local_ep}\n')
    return
