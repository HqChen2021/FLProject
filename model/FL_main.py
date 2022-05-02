import os
import sys
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from options import args_parser
from update import Client, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, plot_dis
from opacus.validators import ModuleValidator
from collections import OrderedDict

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    TrainSet_per_client, TestSet_per_client = get_dataset(args)

    # BUILD MODEL
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
        img_size = TrainSet_per_client[0][0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        print('Error: unrecognized model')
        sys.exit(1)

    errors = ModuleValidator.validate(global_model, strict=False)
    if not errors:
        global_model = ModuleValidator.fix(global_model)
        print(f'\n Revised global model to compatible with DP\n')
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # load dataset and user groups
    TrainSet_per_client, TestSet_per_client = get_dataset(args)
    # fig = plot_dis(TrainSet_per_client)
    pass
    # instantiate client objects
    client = []
    for idx in range(args.num_clients):
        client.append(Client(idx, args, model=copy.deepcopy(global_model),
                             train_set=TrainSet_per_client[idx],
                             test_set=TestSet_per_client[idx]))

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        # global_model.train()
        m = max(int(args.frac * args.num_clients), 1)
        selected_clients = np.random.choice(range(args.num_clients), m, replace=False)
        # FL training
        local_weights, local_losses, local_acc = [], [], []
        for idx in selected_clients:
            w, loss, acc = client[idx].update_model(global_round=epoch, privacy_budget = args.epsilon)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc))
        # update global weights
        global_weights = average_weights(local_weights)

        # replace the '_module' prefix of global_weights'
        new_global_weights = OrderedDict()
        for k,v in global_weights.items():
            name = k[8:]
            new_global_weights[name] = v
        global_model.load_state_dict(new_global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_clients):
            acc, loss = client[idx].inference(model=global_model)
            # game part. if the client isn't satisfied with acc
            # he may need to increase target privacy budget
            # if acc < target_acc
            list_acc.append(round(acc, 2))
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
            print(f'accuracy of each client: {list_acc}')
            
    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, TestSet_per_client)
    test_acc = np.mean([client[i].acc for i in range(args.num_clients)])
    test_loss = np.mean([client[i].loss for i in range(args.num_clients)])

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(test_acc * 100))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    save_path = './save'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_name = os.path.join(save_path, '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
                             format(args.dataset, args.model, args.epochs, args.frac, 
                                    args.iid, args.local_ep, args.local_bs))
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    # plot data distribution via heatmap
    # fig = plot_dis(TrainSet_per_client)