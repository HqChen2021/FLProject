import os

import copy
import time
import pickle
from typing import List, Any

import numpy as np
from tqdm import tqdm
import torch
from options import args_parser
from client import client
import datetime
import logging
from utils import get_dataset, average_weights, exp_details, Initialize_Model, plot_dis
from opacus.validators import ModuleValidator
from collections import OrderedDict

if __name__ == '__main__':
    np.random.seed(1)
    args = args_parser()
    start_time = time.time()
    path_project = os.path.abspath('..')
    save_path = '..\save'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, '{}_{}_C[{}]_F[{}]_iid[{}]_Epoch[{}]_Lep[{}]_B[{}]_DP[{}].npy'.
                             format(args.dataset, args.model, args.num_clients, args.frac,
                                    args.iid, args.epochs, args.local_ep, args.local_bs, args.is_dp))

    # setting log
    # if args.log_file_name is None:
    #     log_file_name = "{}_{}_Gr{}_F{}_Iid{}_Lep{}_Lbs{}".\
    #                         format(args.dataset, args.model, args.epochs, args.frac,
    #                                args.iid, args.local_ep, args.local_bs) + '_%s.json' \
    #                     % datetime.datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
    # log_path = path_project + '\log'
    # if not os.path.exists(log_path):
    #     os.mkdir(log_path)
    # logging.basicConfig(filename=os.path.join(log_path, log_file_name),
    #                     format='%(asctime)s %(levelname)-8s %(message)s',
    #                     datefmt='%m-%d %H:%M', level=logging.DEBUG,
    #                     filemode='w', force=True)
    # logger = logging.getLogger()
    # print experiment detail
    # for arg, value in sorted(vars(args).items()):
    #     logger.info("Argument %s: %r", arg, value)

    exp_details(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    TrainSet_per_client, TestSet_per_client = get_dataset(args)

    # BUILD MODEL
    global_model = Initialize_Model(args)
    errors = ModuleValidator.validate(global_model, strict=False)
    if not errors:
        global_model = ModuleValidator.fix(global_model)
        print((f' Revised global model to compatible with DP'))
        # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, Avg_test_accuracy = [], []
    sample_count = [[], []]

    # load dataset and user groups
    TrainSet_per_client, TestSet_per_client = get_dataset(args)

    sample_count[0] = plot_dis(TrainSet_per_client)
    sample_count[1] = plot_dis(TestSet_per_client)
    # pass
    # instantiate client objects
    client_lst = []
    for idx in range(args.num_clients):
        client_lst.append(client(idx, args, train_set=TrainSet_per_client[idx],
                                 test_set=TestSet_per_client[idx],privacy_budget=args.epsilon))

    best_epoch, best_avg_acc, best_acc_list, state = 0, 0, [], {}
    # FL training
    for epoch in tqdm(range(args.epochs)):
        m = max(int(args.frac * args.num_clients), 1)
        selected_clients = np.random.choice(range(args.num_clients), m, replace=False)
        local_weights, local_losses, local_acc = [], [], []
        for idx in selected_clients:
            w, loss, acc = client_lst[idx].update_model(global_round=epoch,
                                                        model_weights=global_weights)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc))
        if args.is_dp:
            print('Round: {}|\tClient:{}|\tLoss: {}|\tAccuracy: {}|\tε={}'.format(
                epoch, selected_clients,
                [round(n, 3) for n in local_losses],
                [round(n, 3) for n in local_acc],
                [round(client_lst[i].eps[epoch], 3) for i in selected_clients]))
        else:
            print('| Round: {} | Client:{} | Loss: {} | Training Accuracy: {}'.format(
                epoch, selected_clients,
                [round(n, 3) for n in local_losses],
                [round(n, 3) for n in local_acc]))
        # update global weights
        global_weights = average_weights(local_weights)

        # calculate average loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_clients):
            acc, loss = client_lst[idx].inference(global_weights)

            # game part. if the client isn't satisfied with acc
            # he may need to increase target privacy budget
            # if acc < client_lst[idx].target_acc:
            #
            #     client_lst[idx].privacy_budget = 1

            list_acc.append(round(acc, 3))
            list_loss.append(loss)

        Avg_test_accuracy.append(sum(list_acc) / len(list_acc))
        state['Avg_test_acc'] = Avg_test_accuracy
        if Avg_test_accuracy[-1] > best_avg_acc:
            best_avg_acc = Avg_test_accuracy[-1]
            best_acc_list = list_acc
            best_var = np.var(list_acc)
            best_epoch = epoch
            state.update({
                'best_avg_acc': best_avg_acc,
                'best_var': best_var,
                'configures': args,
                'client_lst': client_lst,
                'sample_count': sample_count,
                'state_dict': global_weights,  # 保存模型参数
            })
            torch.save(state, save_path)

        print('Best_acc:{:.2%}, epoch:{:d}, variance:{:.2%} \nacc_lst:{}'.format(best_avg_acc, best_epoch, best_var, best_acc_list))

    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, TestSet_per_client)
    # test_acc = np.mean([client_lst[i].acc for i in range(args.num_clients)])
    # test_loss = np.mean([client_lst[i].loss for i in range(args.num_clients)])

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * Avg_test_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(test_acc * 100))
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # plot data distribution via heatmap
