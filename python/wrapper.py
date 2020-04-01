#!/usr/bin/env/python
import csv
import torch
import time
import os
import numpy as np
import random
from python.utils import CumulativeTripletLoss, SimilarityLoss
from python.data_loader import GraphDataLoader
from python.model import GGNN


class ChemModel(object):

    def __init__(self, log_dir="logs/", directed=True, hidden_size=50, annotation_size=10, edge_types=1,
                 max_nodes=300, timesteps=6, lr=0.001, seed=0, clamp_gradient_norm=1.0):

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        self.log_file = os.path.join(log_dir, "%s_log.csv" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        self.model = GGNN(hidden_size,
                          annotation_size,
                          edge_types,
                          max_nodes,
                          timesteps)

        self.criterion = SimilarityLoss()
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
        accuracy = 0

        for adj_matrix, features, src, mask in data:
            step += 1


            self.optimizer.zero_grad()
            self.model.zero_grad()

            batch_size = adj_matrix.shape[0]
            option_size = adj_matrix.shape[1]
            adj_matrix = adj_matrix.view(-1, adj_matrix.shape[2], adj_matrix.shape[3]).float()
            src = src.view(-1).long()
            features = features.view(-1, features.shape[2], features.shape[3]).float()
            distances = self.model.forward_src(features, adj_matrix, src, batch_size, option_size)

            loss, acc = self.criterion(distances, mask)
            accuracy += acc
            if step % 10 == 0:
                print("epoch {} step {} acc {}".format(epoch, step, acc), flush=True)
            total_loss += np.asscalar(loss.cpu().data.numpy())

            if is_training:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              self.clamp_gradient_norm)
                self.optimizer.step()

        print("Acc: ", accuracy/step)
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
    loader = GraphDataLoader(directory='/Volumes/My Passport/import_prediction/data/graphs/newMethod/',
                             hidden_size=20, directed=False, max_nodes=300, target_edge_type=1)
    test_data = loader.load('test.json', batch_size=10, shuffle=False, targets="generateOnPass")
    # train_data = loader.load("train.json", batch_size=100, shuffle=True, targets="targets_1")
    val_data = loader.load('valid.json', batch_size=10, shuffle=False, targets="targets_1")
    model = ChemModel(log_dir='/Volumes/My Passport/import_prediction/logs/',
                      directed=False,
                      hidden_size=loader.hidden_size,
                      annotation_size=loader.annotation_size,
                      edge_types=loader.edge_types,
                      max_nodes=loader.max_nodes,
                      seed=0,
                      timesteps=4,
                      lr=0.001)
    model.train(epochs=10, patience=2, train_data=test_data, val_data=val_data)
    test_loss = model.run_epoch(val_data, 1, False)
