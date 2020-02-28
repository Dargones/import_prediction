#!/usr/bin/env/python
import csv
import torch
import time
import os
import numpy as np
import random
from torch.autograd import Variable
from utils import LossLessTripletLoss, correct_func
from data_loader import GraphDataLoader
from model import GGNN


class ChemModel(object):

    def __init__(self, log_dir="logs/", directed=True, hidden_size=50, annotation_size=10, edge_types=1,
                 max_nodes=300, timesteps=6, lr=0.001, seed=0, clamp_gradient_norm=1.0):

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        self.log_file = os.path.join(log_dir, "%s_log.csv" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        self.model = GGNN(hidden_size,
                          annotation_size,
                          (edge_types if directed else int(edge_types/2)),
                          max_nodes,
                          timesteps)

        self.criterion = LossLessTripletLoss(dim=hidden_size)
        self.metric = correct_func
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clamp_gradient_norm = clamp_gradient_norm
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def run_epoch(self, data, epoch, is_training):
        """
        XXX: Up to this point, the only thing that needs changing are a few bits in
        process_raw_graphs and train. A couple of things might have to be changed inside here, but
        not too much
        :param data:
        :param epoch:
        :param is_training:
        :return:
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        step = 0
        total_samples = 0
        correct_count = 0

        for adj_matrix, features, targets in data:
            print("epoch {} step {}".format(epoch, step), flush=True)
            step += 1

            features = Variable(features, requires_grad=False)
            adj_matrix = Variable(adj_matrix, requires_grad=False)
            targets = Variable(targets, requires_grad=False)

            total_samples += targets.shape[0]

            self.optimizer.zero_grad()
            self.model.zero_grad()
            anchor, positive, negative = self.model(features, adj_matrix, targets)

            loss = self.criterion(anchor, positive, negative)
            correct_count += self.metric(anchor, positive, negative)
            total_loss += np.asscalar(loss.cpu().data.numpy())

            if is_training:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              self.clamp_gradient_norm)
                self.optimizer.step()

        print("Acc: ", correct_count/total_samples)
        return total_loss


    def train(self, epochs, patience, train_data, val_data):

        best_val_loss, best_val_loss_epoch = float("inf"), 0
        for epoch in range(epochs):
            print("epoch {}".format(epoch))         
            train_loss = self.run_epoch(train_data, epoch, True)
            print("Epoch {} Train loss {}".format(epoch, train_loss))
            val_loss = self.run_epoch(val_data, epoch, False)
            print("Epoch {} Val loss {}".format(epoch, val_loss))

            log_entry = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'valid_loss': val_loss,
                        }

            with open(self.log_file, 'a') as f:
                w = csv.DictWriter(f, log_entry.keys())
                w.writerow(log_entry)

            if val_loss < best_val_loss:
                self.save_model(self.best_model_file)
                print(" (Best epoch so far, cum. val. loss decreased to %.5f from %.5f. "
                      "Saving to '%s')" % (val_loss, best_val_loss, self.best_model_file))
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
            elif epoch - best_val_loss_epoch >= patience:
                print("Stopping training after %i epochs without improvement on "
                      "validation loss." % patience)
                break
 

    def save_model(self, path):
        data_to_save = {"model_weights": self.model.state_dict()}
        torch.save(data_to_save, path)


if __name__ == "__main__":
    loader = GraphDataLoader(directory='data/code/', hidden_size=50, directed=False, max_nodes=250)
    train_data = loader.load("train.json", batch_size=200, new_targets=True, shuffle=True)
    val_data = loader.load('valid.json', batch_size=200, new_targets=True, shuffle=False)
    model = ChemModel(log_dir='logs/',
                      directed=False,
                      hidden_size=loader.hidden_size,
                      annotation_size=loader.annotation_size,
                      edge_types=loader.edge_types,
                      max_nodes=loader.max_nodes,
                      seed=0,
                      timesteps=6,
                      lr=0.001)
    model.train(epochs=100, patience=3, train_data=train_data, val_data=val_data)
