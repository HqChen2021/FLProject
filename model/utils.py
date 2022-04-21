#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
#from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
#from sampling import cifar_iid, cifar_noniid,dir_sampling
from sampling import dir_sampling,iid_sampling

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
    elif args.dataset == 'mnist' :
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif args.dataset == 'fmnist':
        data_dir = './data/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    if args.iid:
        user_groups = iid_sampling(train_dataset, args.num_users)
    else:
        user_groups = dir_sampling(train_dataset, args.num_users, args.alpha)
    return train_dataset, test_dataset, user_groups


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
    print(f'    Total #users  :      \t{args.num_users}')
    print(f'    Fraction of users  : \t{args.frac}')
    print(f'    Local Batch size   : \t{args.local_bs}')
    print(f'    Local Epochs       : \t{args.local_ep}\n')
    return
