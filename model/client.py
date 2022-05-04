#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import Initialize_Model
from collections import OrderedDict


class client(object):
    def __init__(self, idx, args, train_set, test_set):
        self.idx = idx
        self.train_set = train_set
        self.test_set = test_set
        self.args = args
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.criterion = nn.NLLLoss().to(self.device)
        self.DELTA = 0.9 * 1 / len(train_set)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=args.local_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=args.local_bs, shuffle=True)
        self.acc = []
        self.loss = []
        self.eps = {} # keys=round No. values = actual epsilon

    # reference https://github.com/pytorch/opacus/blob/main/tutorials/building_image_classifier.ipynb
    #  Generally, it should be set to be less than
    #  the inverse of the size of the training dataset.
    def load_weights(self, model, weights):
        """
        load model.state_dict with "weights" has prefix like "_module."
        Input: model, weights
        Return: updated model
        """
        new_global_weights = OrderedDict()
        for k, v in weights.items():
            if "_module" not in k:
                model.load_state_dict(weights)
                return model
            else:
                name = k[8:]  # account for prefix "_module."
                new_global_weights[name] = v
        model.load_state_dict(new_global_weights)
        return model

    def update_model(self, global_round, model_weights, privacy_budget):
        epoch_loss, epoch_acc = [], []
        privacy_engine = PrivacyEngine()
        model = Initialize_Model(self.args)
        model = self.load_weights(model, model_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.args.local_ep,
            target_epsilon=privacy_budget,
            target_delta=self.DELTA,
            max_grad_norm=self.args.MAX_GRAD_NORM
        )
        # Set mode to train model
        model.train()
        model.to(self.device)
        with BatchMemoryManager(data_loader=train_loader,
                                max_physical_batch_size=self.args.MAX_PHYSICAL_BATCH_SIZE,
                                optimizer=optimizer) as memory_safe_data_loader:
            for epoch in range(self.args.local_ep):
                batch_loss = []
                batch_acc = []
                for batch_idx, (images, target) in enumerate(memory_safe_data_loader):
                    optimizer.zero_grad()
                    images, target = images.to(self.device), target.to(self.device)
                    # model.zero_grad() the same as optimizer.zero_grad()
                    # calculate batch loss
                    output = model(images).to(self.device)
                    loss = self.criterion(output, target)
                    batch_loss.append(loss.item())
                    # calculate batch accuracy
                    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                    labels = target.detach().cpu().numpy()
                    batch_acc.append((preds == labels).mean().item())
                    loss.backward()
                    optimizer.step()
                epoch_loss.append(np.mean(batch_loss))
                epoch_acc.append(np.mean(batch_acc))
                if self.args.verbose:
                    print('| Global Round: {} | Client:{} | Local Epoch: {} | \tLoss: {:.6f}'
                          '| Accuracy: {}'.format(global_round, self.idx, epoch,
                                                  epoch_loss[-1], epoch_acc[-1]))
        self.eps[global_round] = privacy_engine.get_epsilon(delta=self.DELTA)
        return model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)


    def inference(self, model_weights):
        """
        Input model_weights
        Return inference accuracy and loss.
        """
        model = Initialize_Model(self.args)
        model = self.load_weights(model, model_weights)
        # model.load_state_dict(model_weights)
        model.eval()
        model.to(self.device)
        batch_acc, batch_loss = [], []
        for batch_idx, (images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            output = model(images)
            loss = self.criterion(output, target)
            batch_loss.append(loss.item())
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            batch_acc.append((preds == labels).mean().item())
        self.acc.append(np.mean(batch_acc))
        self.loss.append(np.mean(batch_acc))
        return np.mean(batch_acc), np.mean(batch_loss)
