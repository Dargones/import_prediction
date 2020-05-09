import torch as tt
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import random
from tqdm.auto import tqdm
import sys

from python.utils import CustomTripletLoss
from python.data_loader import GraphDataset
from python.model import GGNN

seed = 0
random.seed(seed)
np.random.seed(seed)
tt.manual_seed(seed)

CUDA = True


def load_dataset(directory, filename, batch_size, shuffle, targets, hidden_size, annotation_size,
                 max_nodes=300, edge_types=3, target_edge_type=1, num_workers=4, max_targets=7):
    """
    Load a .json file into memory, create a GraphDataset out of it and return a DataLoader for it
    :param directory:       the directory from which to load the file
    :param filename:        the name of the file
    :param batch_size:      batch size used to initialize the DataLoader
    :param shuffle:         if True, shuffle the graphs on each pass
    :param targets:         Can be either "generate", "generateOnPass", or a key to the json dictionary
                            from which to load the targets
                            "generate": generate targets once and keep them this way (validation)
                            "generateOnPass": generate new targets at each epoch (training)
    :param hidden_size:     the size of node embedding in GGNN that will be used on this dataset
    :param max_nodes:       maximum number of nodes per graph
    :param edge_types:      number of different edge-types. Does not include the edges added to
                            the undirected graph
    :param annotation_size: the size of annotations (initial embedddings) for each node
    :param target_edge_type:the type of edge that is to be predicted
    :return:                a dataloader object
    """
    full_path = os.path.join(directory, filename) # full path to the file
    print("Loading data from %s" % full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)
    dataset =  GraphDataset(data,
                            hidden_size=hidden_size,
                            max_nodes=max_nodes,
                            edge_types=edge_types,
                            annotation_size=annotation_size,
                            targets=targets,
                            target_edge_type=target_edge_type,
                            max_targets=max_targets)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


def run_epoch(model, optimizer, criterion, data, epoch, is_training, useTqdm=False):
    """
    Run a given model for one epoch on the given dataset and return the loss and the accuracy
    :param data:         a DataLoader object that works on a GraphDataset
    :param epoch:        epoch id (for logging purposes)
    :param is_training:  whether to train or just evaluate
    :param uesTqdm:      use tqdm for output
    :return:             mean loss, mean accuracy
    """
    if CUDA:
        model.cuda()

    if is_training:
        model.train()
        if useTqdm:
            print("Epoch %d. Training" % epoch)
    else:
        model.eval()
        if useTqdm:
            print("Epoch %d. Evaluating" % epoch)

    total_loss = 0
    total_acc = 0
    step = 0
    if useTqdm:
        batches = tqdm(data)
    else:
        batches = data

    for adj_matrix, features, src, mask in batches:
        step += 1

        optimizer.zero_grad()
        model.zero_grad()

        batch_size = adj_matrix.shape[0]
        option_size = adj_matrix.shape[1]

        # TODO: move these view functions in the GraphDataset
        adj_matrix = adj_matrix.view(-1, adj_matrix.shape[2], adj_matrix.shape[3]).float()
        src = src.view(-1).long()
        features = features.view(-1, features.shape[2], features.shape[3]).float()

        if CUDA:  # move to CUDA, if possible
            mask = mask.cuda()
            features = features.cuda()
            adj_matrix = adj_matrix.cuda()
            src = src.cuda()

        distances = model(features, adj_matrix, src, batch_size, option_size)

        loss, acc = criterion(distances, mask)
        total_acc += acc.cpu().data.numpy().item()
        total_loss += loss.cpu().data.numpy().item()

        if useTqdm:
            batches.set_description("Acc: step=%.2f, m.=%.3f. M. loss=%.3f" % (acc, total_acc / step, total_loss / step))

        if is_training:
            loss.backward(retain_graph=True)
            tt.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    if not useTqdm:
        print("Epoch: %d, Acc: %.4f, Loss: %.4f" % (epoch, total_acc/step, total_loss/step))
    return total_loss/step, total_acc/step


def train(model, epochs, optimizer, criterion, patience, train_data, val_data, best_model_file,
          useTqdm=False):
    """
    Train a given model for a given number of epochs. Use early stopping with given patience.
    :param model:     a GGNN model
    :param epochs:    maximum number of epochs to run the model for
    :param patience:  patience value to use for early stopping
    :param train_data:training dataset
    :apram val_data:  validation dataset
    :param best_model_file: file to save the best model to
    :param uesTqdm:   use tqdm for output
    """

    best_val_loss, best_val_loss_epoch = float("inf"), 0
    for epoch in range(epochs):
        run_epoch(model, optimizer, criterion, train_data, epoch, True, useTqdm)
        val_loss, _ = run_epoch(model, optimizer, criterion, val_data, epoch, False, useTqdm)

        if val_loss < best_val_loss:
            tt.save(model.state_dict(), best_model_file)
            if useTqdm:
                print("Best epoch so far. Saving to '%s')" % (best_model_file))
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
        elif epoch - best_val_loss_epoch >= patience:  # early stopping
            print("Early Stopping after %i epochs." % epoch)
            break


def train_test(dim, lr, steps):
    DIR = 'data/graphs/newMethod' + str(dim) + '/'
    test_data = load_dataset(DIR, 'test.json', batch_size=20, shuffle=False, targets="targets_1",
                             hidden_size=dim * 2, annotation_size=dim * 2, max_targets=15,
                             num_workers=1)
    train_data = load_dataset(DIR, "train.json", batch_size=20, shuffle=True,
                              targets="generateOnPass", hidden_size=dim * 2, annotation_size=dim * 2,
                              max_targets=15, num_workers=1)
    val_data = load_dataset(DIR, 'valid.json', batch_size=20, shuffle=False, targets="generate",
                            hidden_size=dim * 2, annotation_size=dim * 2, max_targets=15,
                            num_workers=1)

    model = GGNN(state_dim=dim * 2,
                 annotation_dim=dim * 2,
                 n_edge_types=3,
                 n_nodes=300,
                 n_steps=steps)
    criterion = CustomTripletLoss(margin=0.5)
    optimizer = tt.optim.Adam(model.parameters(), lr=lr)

    best_filename = "best"+str(dim)+"-" + str(lr) + "_" + str(steps) + ".model"

    train(model, epochs=30, optimizer=optimizer, criterion=criterion, patience=3,
          train_data=train_data, val_data=val_data, best_model_file=best_filename, useTqdm=False
          )

    model.load_state_dict(tt.load(best_filename))
    loss, acc = run_epoch(model, optimizer, criterion, test_data, 1, False, useTqdm=False)
    print(loss, acc)


def test_model(dim, steps, candidates, model_file):
    DIR = '../data/graphs/newMethod' + str(dim) + '/'
    test_data = load_dataset(DIR, 'test.json', batch_size=1, shuffle=False,
                             targets="targets_" + str(candidates), hidden_size=dim * 2,
                             annotation_size=dim * 2, num_workers=1, max_targets=candidates)

    model = GGNN(state_dim=dim * 2,
                 annotation_dim=dim * 2,
                 n_edge_types=3,
                 n_nodes=300,
                 n_steps=steps)
    criterion = CustomTripletLoss(margin=0.5, binary_acc=False)
    optimizer = tt.optim.Adam(model.parameters(), lr=0.001)

    model.load_state_dict(tt.load(model_file))
    loss, acc = run_epoch(model, optimizer, criterion, test_data, 1, False, useTqdm=True)
    print(loss, acc)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_test(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "test":
        test_model(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])


