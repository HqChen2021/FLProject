#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
class Client(object):
    def __init__(self, idx, args,  model, train_set, test_set):
        self.idx = idx
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.args = args
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.criterion = nn.NLLLoss().to(self.device)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5)
        self.DELTA = 0.9 * 1 / len(train_set)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.local_bs, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                        batch_size=args.local_bs, shuffle=True)
        self.acc = []
        self.loss = []
# reference https://github.com/pytorch/opacus/blob/main/tutorials/building_image_classifier.ipynb
#  Generally, it should be set to be less than
#  the inverse of the size of the training dataset.

    def update_model(self, global_round, privacy_budget):
        epoch_loss, epoch_acc = [], []
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.args.local_ep,
            target_epsilon=privacy_budget,
            target_delta=self.DELTA,
            max_grad_norm=self.args.MAX_GRAD_NORM
        )
        # Set mode to train model
        model.train()
        with BatchMemoryManager(data_loader=train_loader,
                                max_physical_batch_size=self.args.MAX_PHYSICAL_BATCH_SIZE,
                                optimizer=optimizer) as memory_safe_data_loader:
            for iter in range(self.args.local_ep):
                batch_loss = []
                batch_acc = []
                for batch_idx, (images, target) in enumerate(memory_safe_data_loader):
                    optimizer.zero_grad()
                    images, target = images.to(self.device), target.to(self.device)
                    # model.zero_grad() the same as optimizer.zero_grad()
                    # calculate batch loss
                    output = model(images)
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
                          '| Accuracy: {}'.format(
                        global_round, self.idx, iter, epoch_loss[-1], epoch_acc[-1]))
        return model.state_dict(), np.mean(epoch_loss), np.mean(epoch_acc)
        # eps = privacy_engine.get_epsilon(delta=self.args.target_delta)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        batch_acc, batch_loss = [], []
        for batch_idx, (images, target) in enumerate(self.test_loader):
            images, target = images.to(self.device), target.to(self.device)
            # Inference
            output = model(images)
            loss = self.criterion(output, target)
            batch_loss.append(loss.item())
            # batch_loss = self.criterion(outputs, labels)
            # loss += batch_loss.item()
            # Prediction
            # _, pred_labels = torch.max(outputs, 1)
            # pred_labels = pred_labels.view(-1)
            # correct += torch.sum(torch.eq(pred_labels, target)).item()
            # total += len(target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            batch_acc.append((preds == labels).mean().item())
        self.acc.append(np.mean(batch_acc))
        self.loss.append(np.mean(batch_acc))
        return np.mean(batch_acc),np.mean(batch_loss)


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    criterion = nn.NLLLoss().to(device)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False)

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
