# A module for loading the graph data from .json files and preparing them for input to GGNNs

import numpy as np
import torch as tt
from torch.utils.data import Dataset, DataLoader
import random
import os
import json

class GraphDataLoader:
    """
    A class that allows loading multiple parts of the same dataset. The class makes sure that the
    number of nodes, types of edges, and the size of annotations is consistent across the dataset
    """

    def __init__(self, directory, hidden_size, directed=False, max_nodes=None, edge_types=None, annotation_size=None):
        """
        Initialize the GraphDataLoader and set some of the parameters that should be consistent
        across all parts of the dataset
        :param directory:       the directory from which to load the files
        :param hidden_size:     the size of node embedding in GGNN that will be used on this dataset
        :param directed:        whether the graph is directed. If directed=False, each edge will be
                                accompanied with an additional edge going in another direction
        :param max_nodes:       maximum number of nodes per graph
        :param edge_types:      number of different edge-types. Does not include the edges added to
                                the undirected graph
        :param annotation_size: the size of annotations (initial embedddings) for each node
        """
        self.directory = directory
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size

    def load(self, filename, batch_size, shuffle=False, targets="generateOnPass"):
        """
        Method that loads a .json file with graphs to memory and returns a DataLoader that can
        be used to feed the graphs to the neural network.
        :param filename:        name of the .json file with graphs
        :param batch_size:      batch size used to initialize the DataLoader
        :param shuffle:         if True, shuffle the graphs on each pass
        :param targets:     Can be either "generate", "generateOnPass", or "preset"
                                "generate": generate targets once and keep them this way (valid)
                                "generateOnPass": generate new targets each epoch (train)
                                "preset": use the targets specified in the file
        :return: a DataLoader object
        """
        full_path = os.path.join(self.directory, filename) # full path to the file
        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        self.update_parameters(data)  # make sure that all parameters are consistent across the data
        dataset =  GraphDataset(data,
                                hidden_size=self.hidden_size,
                                directed=self.directed,
                                max_nodes=self.max_nodes,
                                edge_types=self.edge_types,
                                annotation_size=self.annotation_size,
                                targets=targets)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=2)

    def update_parameters(self, data):
        """
        Given a new dataset, update such parameters as the maximum number of nodes and edge
        types if these parameters have not been set in advance. If the parameters have already
        been initialized, assert that the parameters match
        :param data: Raw .json dataset
        :return: None
        """
        edge_types_tmp = 0
        max_nodes_tmp = 0
        annotation_size_tmp = len(data[0]["annotations"][0])

        for graph in data:
            max_nodes_tmp = max(max_nodes_tmp, len(graph["annotations"]))
            edge_types_tmp = max(edge_types_tmp, max([e[1] for e in graph['edges']]))

        if not self.edge_types:
            self.edge_types = edge_types_tmp
        if not self.max_nodes:
            self.max_nodes = max_nodes_tmp
        if not self.annotation_size:
            self.annotation_size = annotation_size_tmp

        assert(edge_types_tmp <= self.edge_types)
        assert(max_nodes_tmp <= self.max_nodes)
        for graph in data:
            assert(len(graph["annotations"][0]) == annotation_size_tmp)


class GraphDataset(Dataset):
    """
    This class allows converting graphs to their neural-network representations on demand
    """

    def __init__(self, data, hidden_size, directed, max_nodes, edge_types, annotation_size, targets):
        """
        Initialize GraphDataset so that it can be passed to DataLoader
        :param data:            graph data in .json format as loaded from disk
        :param hidden_size:     the size of node embedding in GGNN that will be used on this dataset
        :param directed:        whether the graph is directed. If directed=False, each edge will be
                                accompanied with an additional edge going in another direction
        :param max_nodes:       maximum number of nodes per graph
        :param edge_types:      number of different edge-types. Does not include the edges added to
                                the undirected graph
        :param annotation_size: the size of annotations (initial embedddings) for each node
        :param targets:         Can be either "generate", "generateOnPass", or "preset"
                                "generate": generate targets once and keep them this way (valid)
                                "generateOnPass": generate new targets each epoch (train)
                                "preset": use the targets specified in the file
        """
        self.data = data
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size
        self.targets = targets
        # if targets are ot be automatically generated, clear whatever is stored as targets now:
        if self.targets != "preset":
            for graph in self.data:
                graph['targets'] = None

    def __len__(self):
        """
        Return the number of samples in the dataset
        :return:
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get the data given its index in the dataset.
        :param index:   the index by which a graph can be identified in the dataset
        :return:        A tuple of three torch tensors:
                        The first one is the adjacency matrix,
                        The second contains node features,
                        The third contains the targets
        """
        graph = self.data[index]
        a_matrix = self.create_adjacency_matrix(graph["edges"])  # adjacency matrix
        # node features should have the the shape (Max#OfNodes, hidden_size)
        features = np.pad(graph['annotations'],
                          ((0, self.max_nodes - len(graph['annotations'])),
                           (0, self.hidden_size - self.annotation_size)),
                          'constant')

        if self.targets == "generateOnPass" or (self.targets == "generate" and not graph["targets"]):
            graph["targets"] = self.create_target(graph['edges'], a_matrix)
        src, pos, mask = graph["targets"]
        return tt.FloatTensor(a_matrix), tt.FloatTensor(features), tt.LongTensor(mask), src, pos

    def create_adjacency_matrix(self, edges):
        """
        Create adjacency matrix for the graph
        :param edges: List of all edges in the graph
        :return:
        """
        a = np.zeros([self.max_nodes, self.max_nodes * self.edge_types * (2 - self.directed)])
        for edge in edges:
            src = edge[0]
            e_type = edge[1]
            dest = edge[2]
            a[dest - 1][(e_type - 1) * self.max_nodes + src - 1] = 1  # a[target][source]
            if self.directed:
                continue
            a[src - 1][(e_type - 1 + self.edge_types) * self.max_nodes + dest - 1] = 1
        return a

    def create_target(self, edges, a):
        """
        Modify the graph by removing an edge. Return a triplet of node indices that are to be used
        later during loss function calculation
        :return:
        """
        src, e_type, dest = edges[random.randint(0, len(edges) - 1)]  # select a random edge
        # a column that for each node specifies whether there is an edge to it from the source node:
        mask = 1 - np.resize(a[:, (e_type - 1) * self.max_nodes + src - 1], self.max_nodes)
        mask[dest - 1] = 0  # remove positive example from the mask
        a[dest - 1][(e_type - 1) * self.max_nodes + src - 1] = 0

        if not self.directed:
            a[src - 1][(e_type - 1 + self.edge_types) * self.max_nodes + dest - 1] = 0

        return src - 1, dest - 1, mask