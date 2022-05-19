import copy
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import MLP, CNN, CNN, CNNCifar
import sys
import os


def Initialize_Model(args):
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNN(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNN(args=args)
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
        global_model = MLP(dim_in=len_in, dim_hidden=500,
                           dim_out=args.num_classes)
    else:
        print('Error: unrecognized model')
        sys.exit(1)
    return global_model


def dir_sampling(train_set, test_set, num_clients, alpha=0.5):
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
    num_classes = len(train_set.classes)
    num_all_data = len(train_set)  # dataset.shape = 60000,28,28 for FMNIST
    client_data_idx_map = {}
    least_samples = 50
    while min_size < least_samples:
        train_data_index = [[] for _ in range(num_clients)]
        test_data_index = [[] for _ in range(num_clients)]
        # data_index is a list stores data_index (which is also a list)
        # for clients. initialized as empty and increases step by step
        # equivalent to a sub-dataset
        for k in range(num_classes):
            train_idx_of_class_k = np.where(train_set.targets == k)[0]
            test_idx_of_class_k = np.where(test_set.targets == k)[0]
            # locate index belong to data from class k
            np.random.shuffle(train_idx_of_class_k)
            np.random.shuffle(test_idx_of_class_k)
            # introduce randomness?
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # generate sampling probabilities, [p_1,p_2,...p_K] \sum_{i=1}^{K}(p_i)=1
            proportions = np.array([p * (len(idx_j) < num_all_data / num_clients)
                                    for p, idx_j in zip(proportions, train_data_index)])
            # check the probabilities list, if client j has gain enough samples,
            # then his sampling probability is set to 0 for this round and afterward
            proportions = proportions / proportions.sum()
            # resize the prob list
            train_proportions = (np.cumsum(proportions) * len(train_idx_of_class_k)).astype(int)[:-1]
            test_proportions = (np.cumsum(proportions) * len(test_idx_of_class_k)).astype(int)[:-1]
            # calculate the cumulative sum of proba list, rule out the last element
            train_data_index = [idx_j + idx.tolist() for idx_j, idx in
                                zip(train_data_index, np.split(train_idx_of_class_k, train_proportions))]
            test_data_index = [idx_j + idx.tolist() for idx_j, idx in
                               zip(test_data_index, np.split(test_idx_of_class_k, test_proportions))]
            # update data_index, distribute samples from class k accorss all clients
        min_size = min([len(idx_j) for idx_j in train_data_index])
        # calulate min_size to see whether need to re-sampling

    for j in range(num_clients):
        np.random.shuffle(train_data_index[j])
        client_data_idx_map[j] = [train_data_index[j], test_data_index[j]]

    return client_data_idx_map


def iid_sampling(train_set, test_set, num_clients):
    """
    IID sampling
    input: dataset, num_clients
    return: IID samples stored in a dictionary, keys = client id, values = sample index
    param:
    dataset--training dataset
    """
    train_per_clients = int(len(train_set) / num_clients)
    test_per_clients = int(len(test_set) / num_clients)
    # dividing dataset into num_clients parts equally
    client_data_idx_map = {}
    train_all_index, test_all_index = [i for i in range(len(train_set))], \
                                      [i for i in range(len(test_set))]
    for i in range(num_clients):
        # sampling num_items items from dataset, with no repeat
        train_data_index = np.random.choice(train_all_index, int(train_per_clients), replace=False)
        test_data_index = np.random.choice(test_all_index, int(test_per_clients), replace=False)
        client_data_idx_map[i] = [train_data_index, test_data_index]
        # a single sampling will not repeat, however second sampling will
        # coincide with the first at probability, so should update the
        # dataset after each sampling
        train_all_index = list(set(train_all_index) - set(train_data_index))
        test_all_index = list(set(test_all_index) - set(test_data_index))
    return client_data_idx_map


# fig = plot_dis(Train_set_per_client)

def get_dataset(args, data_path):
    """
    returns two list of list. "Train_set_per_client" and "Test_set_per_client"
    Train_set_per_client[i] is the training set of client i
    """
    if args.dataset == 'cifar':
        data_dir = os.path.join(data_path, './cifar/')
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
    elif args.dataset == 'mnist':
        data_dir = os.path.join(data_path, './mnist/')
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    elif args.dataset == 'fmnist':
        data_dir = os.path.join(data_path, './fmnist/')
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True,
                                          transform=apply_transform)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True,
                                         transform=apply_transform)
    if args.iid:
        # client_data_idx_map is a dictionary
        # keys=client index, values = list of sample index
        client_data_idx_map = iid_sampling(train_set, test_set, args.num_clients)
    else:
        client_data_idx_map = dir_sampling(train_set, test_set, args.num_clients, args.alpha)

    # Train_set_per_client & Test_set_per_client is a dictionary
    # keys=client index, values = sub-dataset
    Train_set_per_client, Test_set_per_client = {}, {}
    for idx in range(args.num_clients):
        Train_set_per_client[idx] = torch.utils.data.Subset(
            train_set, client_data_idx_map[idx][0])
        Test_set_per_client[idx] = torch.utils.data.Subset(
            test_set, client_data_idx_map[idx][1])
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
        # this is not working right. Access a subset 1. directly through index, 0,1,...,len(subset)
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
    plt.show(block=True)
    return count_matrix


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print('    Federated parameters:')
    print('    IID') if args.iid else print('    Non-IID')
    print(f'    Use local best') if args.use_local_best else print(f'    Use global model')
    print(f'    Total #users  :      \t{args.num_clients}')
    print(f'    Fraction of users  : \t{args.frac}')
    print(f'    Local Batch size   : \t{args.local_bs}')
    print(f'    Local Epochs       : \t{args.local_ep}\n')

    print('    DP parameters:')
    print(f'    DP mechanism is running') if args.is_dp else print(f'    No DP mechanism is running')
    print(f'    max_grad_norm  : \t{args.max_grad_norm}')
    print(f'    SE game is running') if args.se_game else print(f'    No SE game is running')
    print(f'    min_z  : \t{args.min_z}')
    print(f'    Noise_multiplier:\t{args.noise_multiplier}')
