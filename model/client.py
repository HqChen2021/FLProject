#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
# import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
# from opacus import PrivacyEngine
# from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import Initialize_Model
from collections import OrderedDict
from DPOptimizerClass import DPOptimizer
from opacus.validators import ModuleValidator
from opacus import GradSampleModule


def creat_model(args):
    if args.is_dp:
        model = GradSampleModule(Initialize_Model(args))
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
    else:
        model = Initialize_Model(args)
    return model


class client(object):
    def __init__(self, cid, args, train_set, test_set):
        self.best_acc = None
        self.best_loss = None
        self.cid = cid
        self.args = args
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.criterion = nn.NLLLoss().to(self.device)
        self.delta = 1 / (1.1 * len(train_set))
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.local_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.local_bs, shuffle=True)
        self.test_acc, self.test_loss, self.z = [], [], {}
        self.local_best_round, self.best_acc = 0, 0
        self.target_acc = args.target_acc
        self.model = creat_model(args)
        self.is_se = False
        hasattr(next(self.model.parameters()), 'grad_sample')
        self.noise_multiplier = self.args.noise_multiplier

    def load_weights(self, model, weights):
        """
        load model weights
        """
        for (k1, _), (k2, _) in zip(weights.items(), model.state_dict().items()):
            if k1 == k2:
                model.load_state_dict(weights)
            elif "_module" in k2:
                model._module.load_state_dict(weights)
            else:
                new_global_weights = OrderedDict()
                for k, v in weights.items():
                    name = k[8:]
                    new_global_weights[name] = v
                model.load_state_dict(new_global_weights)
            return model

    def draw_new_z(self, current_z, z_min):
        # pdf f(z)=-2*z/(z_max-current_z)^2+2/(z_max-current_z)
        # using inversion method, starts with u~U(0,1), gengerate z=F^(-1)(u)
        # invers func: lambda y: (z_max - current_z) * (1 - np.sqrt(1 - y))
        return (current_z - z_min) * (1 - np.sqrt(1 - np.random.uniform(0, 1)))

    def whether_change(self, prob):
        """
        args
        prob \in (0,1) : the probability of decrease noise_multiplier,
        determined by acc and participating times

        return
        p(x<prob) where x~U(0,1)
        """
        return True if 0.5*np.random.uniform(0, 1) < prob else False

    def calculate_prob(self, times, acc):
        """
        args
        x negetively related to participated times, the bigger x is, client is less participated
        in training, thus the smaller prob to decrease noise_multiplier. Since it is more likely
        client's data has not been seen by the model yet, and is not appropriate to decrease noise
        too early-->p(x)=(- 2 * x / expect_times**2 + 2 / expect_times)
        acc is the testing accuracy, the smaller acc is, the more likey to decrease noise_multiplier
        --> p(y)=(- 2 * acc / self.args.target_acc**2 + 2 / self.args.target_acc)

        return: rescaled p(x)*p(y), i.e., p(x)/p(x)_max * p(y)/p(y)_max
        """
        expect_times = self.args.epochs * self.args.frac
        x = expect_times - times
        return min((-x / expect_times + 1),1) * np.abs(
            -acc / self.args.target_acc + 1)  # abs account for case where acc > target_acc

    def train(self, model, optimizer, dataloader):
        model.train()
        epoch_loss, epoch_acc = [], []
        for epoch in range(self.args.local_ep):
            batch_loss, batch_acc = [], []
            for batch_idx, (images, target) in enumerate(dataloader):
                images, target = images.to(self.device), target.to(self.device)
                output = model(images)
                loss = self.criterion(output, target)
                batch_loss.append(loss.item())
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                batch_acc.append((preds == labels).mean().item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss.append(np.mean(batch_loss))
            epoch_acc.append(np.mean(batch_acc))
        print('--------------------------------------------')
        return model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)

    def update_model(self, global_round, global_weights):
        test_acc, _ = self.inference(global_weights, infer_with_local_best = False)
        # 判断是否达到SE，是否需要使用本地最优的模型. 在测试dpsgd的时候 is_se取false
        if self.is_se and self.args.use_local_best:
            if test_acc >= self.best_acc:
                self.model = self.load_weights(self.model, global_weights)
                print(f'\nclient {self.cid} increase from {self.best_acc:}(old SE) to {test_acc}(new SE)')
                self.best_acc = test_acc
            else:
                # use self.model in this update_model round, no need to update self.model to global_weights here
                print(f'\nclient {self.cid} with test_acc {self.best_acc} use optimal model in this round')
            # client is SE, won't decrease z anymore
            self.z[global_round] = self.noise_multiplier
        else:
            # 更新本地模型
            self.model = self.load_weights(self.model, global_weights)
        # game part
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.1)
        if self.args.is_dp :
            # 在测试dpsgd的时候 se_game取false
            if self.args.se_game:
                # 如果玩 satisfaction game, 每个client先计算当前执行动作的概率prob
                # prob 由 参与FL的次数len(self.z) 和 当前的 test_acc （global model在本地测试集上的正确率） 决定
                prob = self.calculate_prob(len(self.z), test_acc)
                assert 0 <= prob < 1, "probability is not right"
                # 如果whether_change(prob)=True 且 test_acc 小于 target_acc:
                if self.whether_change(prob) and test_acc < self.target_acc:
                    # client 按照一定概率确定 noise_multiplier 要减小的幅度
                    self.noise_multiplier -= self.draw_new_z(current_z=self.noise_multiplier, z_min=self.args.min_z)
                    last_round = list(self.z.keys())[-1]
                    print(f'\nclient {self.cid} participated in {len(self.z)} times,'
                          f'now decrease z from {self.z[last_round]} to {self.noise_multiplier}')
                # 如果实现了target accuracy 则标记 is_se = True
                elif test_acc > self.target_acc:
                    if self.args.use_local_best:
                        self.is_se = True
                        self.best_acc = test_acc
                        self.model = self.load_weights(self.model, global_weights)
                        print(f'update local optimal model with new acc: {test_acc}')
                    else:
                        # self.z 记录 client每次参与FL学习时采取的 noise_multiplier
                        self.z[global_round] = self.noise_multiplier
                        print(f'\nclient {self.cid} get accuracy: {test_acc} > target_accuracy: {self.args.target_acc} ')
            # 定义了DPOptimizer类， 在其中实现
            # 1. clip 取出模型中的梯度，截断 然后求和
            # 2. 加噪声
            # 3. 除以 L （https://arxiv.org/abs/1607.00133) 算法1 中倒数第五行

            dp_optimizer = DPOptimizer(optimizer=optimizer,
                                       noise_multiplier=self.noise_multiplier,
                                       max_grad_norm=self.args.max_grad_norm,
                                       expected_batch_size=self.args.local_bs)
            # 训练的结果包含 train_results = (模型参数， 正确率均值， loss均值)
            train_results = self.train(self.model, dp_optimizer, self.train_loader)
            # 在测试dpsgd的时候 is_se取false
            self.z[global_round] = self.noise_multiplier
        else:
            train_results = self.train(self.model, optimizer, self.train_loader)
        return train_results

    def inference(self, model_weights, infer_with_local_best = True):
        if infer_with_local_best and self.is_se and self.args.use_local_best:
            model = self.model
            print(f'client[{self.cid} use local best model, with test acc {self.best_acc }')
        else:
            model = Initialize_Model(self.args)
            model = self.load_weights(model, model_weights)
        model.eval()
        batch_acc, batch_loss = [], []
        for batch_idx, (images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            output = model(images)
            loss = self.criterion(output, target)
            batch_loss.append(loss.item())
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            batch_acc.append((preds == labels).mean().item())
        self.test_acc.append(np.mean(batch_acc))
        self.test_loss.append(np.mean(batch_acc))
        return np.mean(batch_acc), np.mean(batch_loss)
