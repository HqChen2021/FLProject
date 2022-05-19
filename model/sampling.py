import numpy as np
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt


def dir_sampling(dataset, num_clients, alpha=0.5):
    """
    dirichlet_distribution_sampling
    input: dataset, num_clients, alpha
    return: dirichlet distributed samples stored in a dictionary, keys = client id, values = sample index
    param:
    dataset--training dataset, i.e., data = datasets.FashionMNIST(root='data/FMNIST',download=True,train=True)
    alpha--controls the dirichlet distribution, is a list of the same length as #classes, could be equal or unequal,
           i.e., alpha = [1,1,1,1] or [1,1,100,1] for class=4, the later will heavily concentrate distribution on class 3
    """
    min_size = 0
    num_classes = len(dataset.classes)
    num_all_data = dataset.data.shape[0] # dataset.shape = 60000,28,28 for FMNIST
    client_dataidx_map = {}
    least_samples = 10
    while min_size < least_samples:
        data_index = [[] for _ in range(num_clients)]
        # data_index is a list stores data_index for clients. initialized as empty and increases step by step
        for k in range(num_classes):
            idx_of_class_k = np.where(dataset.targets == k)[0]
            # locate index belong to data from class k
            np.random.shuffle(idx_of_class_k)
            # introduce randomness?
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # generate sampling probabilities, [p_1,p_2,...p_K] \sum_{i=1}^{K}(p_i)=1
            proportions = np.array([p * (len(idx_j) < num_all_data / num_clients) for p, idx_j in zip(proportions,
                                                                                                      data_index)])
            # check the probabilities lsit, if client j has gain enough samples,
            # tthen his sampling probability is set to 0 for this round and afterward
            proportions = proportions / proportions.sum()
            # resize the prob list
            proportions = (np.cumsum(proportions) * len(idx_of_class_k)).astype(int)[:-1]
            # calculate the cumulative sum of proba list, rule out the last element(which i think is unnecessary)
            data_index = [idx_j + idx.tolist() for idx_j, idx in zip(data_index, np.split(idx_of_class_k, proportions))]
            # update data_index, distribute samples from class k accorss all clients
        min_size = min([len(idx_j) for idx_j in data_index])
        # calulate min_size to see whether need to re-sampling

    for j in range(num_clients):
        np.random.shuffle(data_index[j])
        client_dataidx_map[j] = data_index[j]

    return client_dataidx_map


def iid_sampling(dataset, num_clients):
    """
    IID sampling
    input: dataset, num_clients
    return: IID samples stored in a dictionary, keys = client id, values = sample index
    param:
    dataset--training dataset, i.e., data = datasets.FashionMNIST(root='data/FMNIST',download=True,train=True)
    """
    num_items = int(len(dataset) / num_clients)
    # NOTE dividing dataset into num_users parts equally
    client_dataidx_map, all_index = {}, [i for i in range(len(dataset) )]
    for i in range(num_clients):
        client_dataidx_map[i] = set(np.random.choice(all_index, int(num_items),
                                                     replace=False))
        # sampling num_items items from dataset, with no repeat
        all_index = list(set(all_index) - client_dataidx_map[i])
        # a single sampling will not repeat, however second sampling will
        # coincide with the first at probability, so should update the
        # dataset after each sampling
    return client_dataidx_map


def plot_dis(dataset, client_dataidx_map):
    """
    param:
    dataset
    client_dataidx_map: a dictionary, keys: client id, value: sample indices
    return: a heatmap figure plot by seaborn
    """
    num_clients = len(client_dataidx_map.keys())
    num_classes = len(dataset.classes)
    labels = [[] for _ in range(num_clients)]
    for i in range(10):
        for index in client_dataidx_map[i]:
            _, lable = dataset[index]
            labels[i].append(lable)

    count_matrix = [[] for _ in range(num_clients)]
    for client_idx in range(num_clients):
        count_matrix[client_idx] = [labels[client_idx].count(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 10))
    figure = sns.heatmap(count_matrix, annot=True, annot_kws={'size': 10},
                         fmt='.20g', cmap='Greens', ax=ax)
    figure.set(xlabel='class index', ylabel='client index')
    return figure


# def mnist_iid(dataset, num_users):
#     """
#     sampling i.i.d. client data from MNIST dataset
#     :param dataset, num_users
#     :return dict or image index
#     """
#     num_items = int(len(dataset) / num_users)
#     # NOTE dividing dataset into num_users parts equally
#     client_dataidx_map, all_index = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         client_dataidx_map[i] = set(np.random.choice(all_index, int(num_items),
#                                              replace=False))
#         # sampling num_items items from dataset, with no repeat
#         all_index = list(set(all_index) - client_dataidx_map[i])
#         # a single sampling will not repeat, however second sampling will
#         # coincide with the first at probability, so should update the
#         # dataset after each sampling
#     return client_dataidx_map
#
# def mnist_noniid(dataset, num_users):
#     """
#     sampling Non-i.i.d. client data from MNIST dataset
#     :param dataset, num_users
#     :return dict or image index
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     client_dataidx_map = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     # np.arange(3,7,2) --> array([3, 5])
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     # Stack arrays in sequence vertically (row wise).
#     # a = np.array([1, 2, 3]) b = np.array([4, 5, 6]) np.vstack((a,b)) --> array([[1, 2, 3], [4, 5, 6]])
#     # see https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # index_labels[1:], 0 is idex, 1 is labels
#     # Returns the indices that would sort an array
#     # x = np.array([3, 1, 2]), np.argsort(x) --> array([1, 2, 0]) the indeices
#     # see https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
#     idxs = idxs_labels[0, :]
#
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             client_dataidx_map[i] = np.concatenate(
#                 (client_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return client_dataidx_map
#
# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     client_dataidx_map = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30
#
#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard + 1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)
#
#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:
#
#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 client_dataidx_map[i] = np.concatenate(
#                     (client_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
#                     axis=0)
#
#         random_shard_size = random_shard_size - 1
#
#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 client_dataidx_map[i] = np.concatenate(
#                     (client_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
#                     axis=0)
#     else:
#
#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 client_dataidx_map[i] = np.concatenate(
#                     (client_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
#                     axis=0)
#
#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(client_dataidx_map, key=lambda x: len(client_dataidx_map.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 client_dataidx_map[k] = np.concatenate(
#                     (client_dataidx_map[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
#                     axis=0)
#
#     return client_dataidx_map
#
# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset) / num_users)
#     client_dataidx_map, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         client_dataidx_map[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - client_dataidx_map[i])
#     return client_dataidx_map
#
#
# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 250
#     idx_shard = [i for i in range(num_shards)]
#     client_dataidx_map = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(dataset.train_labels)
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             client_dataidx_map[i] = np.concatenate(
#                 (client_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return client_dataidx_map


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
