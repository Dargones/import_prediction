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

    def __init__(self, directory, hidden_size, directed=False, max_nodes=None, edge_types=None, annotation_size=None, target_edge_type=1):
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
        :param target_edge_type:the type of edge that is to be predicted
        """
        self.directory = directory
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size
        self.target_edge_type = target_edge_type

    def load(self, filename, batch_size, shuffle=False, targets="generateOnPass"):
        """
        Method that loads a .json file with graphs to memory and returns a DataLoader that can
        be used to feed the graphs to the neural network.
        :param filename:        name of the .json file with graphs
        :param batch_size:      batch size used to initialize the DataLoader
        :param shuffle:         if True, shuffle the graphs on each pass
        :param targets:         Can be either "generate", "generateOnPass", or "preset"
                                "generate": generate targets once and keep them this way (valid)
                                "generateOnPass": generate new targets each epoch (train)
                                some other string that is the key to a target field in the data
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
                                targets=targets,
                                target_edge_type=self.target_edge_type)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=4)


    def initialize(self, data, batch_size, shuffle=False, targets="generateOnPass"):
        """
        Method that gets the contents of a json file and returns a data loader to work with
        :param data:            data loaded from json
        :param batch_size:      batch size used to initialize the DataLoader
        :param shuffle:         if True, shuffle the graphs on each pass
        :param targets:         Can be either "generate", "generateOnPass", or "preset"
                                "generate": generate targets once and keep them this way (valid)
                                "generateOnPass": generate new targets each epoch (train)
                                some other string that is the key to a target field in the data
        :return: a DataLoader object
        """
        self.update_parameters(data)  # make sure that all parameters are consistent across the data
        dataset =  GraphDataset(data,
                                hidden_size=self.hidden_size,
                                directed=self.directed,
                                max_nodes=self.max_nodes,
                                edge_types=self.edge_types,
                                annotation_size=self.annotation_size,
                                targets=targets,
                                target_edge_type=self.target_edge_type)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=4)

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

    def __init__(self, data, hidden_size, directed, max_nodes, edge_types, annotation_size, targets, target_edge_type, max_targets=14):
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
                                some other string that is the key to a target field in the data
        :param max_targets:     Maximum number of possible target options
        :param target_edge_type:the type of edge that is to be predicted
        """
        self.data = data
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size
        self.targets = targets
        self.target_edge_type = target_edge_type
        self.max_targets = max_targets
        # if targets are ot be automatically generated, clear whatever is stored as targets now:
        if self.targets == "generate" or self.targets == "generateOnPass":
            for graph in self.data:
                graph['targets'] = None
        else:
            for graph in self.data:
                graph['targets'] = self.read_targets(graph)

    def __len__(self):
        """
        Return the number of samples in the dataset
        :return:
        """
        return len(self.data)

    def __getitem__(self, index):
        return self.getitem_complex(index)

    def getitem_simple(self, index):
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

        if self.targets == "generateOnPass" or (
                self.targets == "generate" and not graph["targets"]):
            graph["targets"] = self.create_target(graph['edges'], a_matrix,
                                                  len(graph['annotations']))
        src, pos, mask = graph["targets"]

        self.set_matrix(a_matrix, src, pos, self.target_edge_type, 0)
        return a_matrix, features, mask, src, pos

    def getitem_complex(self, index):
        matrix, features, mask, src, pos = self.getitem_simple(index)
        options = np.where(mask == 1)[0]
        if len(options) > self.max_targets:
            options = np.random.choice(options, self.max_targets)

        matrixes = np.zeros((self.max_targets + 2, matrix.shape[0], matrix.shape[1]))
        matrixes[0] = matrix

        for i, option in enumerate([pos] + list(options)):
            new_matrix = matrix.copy()
            self.set_matrix(new_matrix, src, option, self.target_edge_type, 1)
            matrixes[i + 1] = new_matrix

        src = np.full(shape=self.max_targets + 2, fill_value=src)
        features = np.stack(list([features for _ in range(self.max_targets + 2)]), axis=0)
        mask = np.zeros(self.max_targets + 2)
        mask[2:len(options) + 2] = 1
        return matrixes, features, src, mask

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
            self.set_matrix(a, src, dest, e_type, 1)
        return a

    def set_matrix(self, a, src, dest, e_type, value):
        a[dest][(e_type - 1) * self.max_nodes + src] = value
        if not self.directed:
            a[src][(e_type - 1 + self.edge_types) * self.max_nodes + dest] = value

    def create_target(self, edges, a, n_nodes):
        """
        Modify the graph by removing an edge. Return a triplet of node indices that are to be used
        later during loss function calculation
        :return:
        """
        valid_edges = [x for x in edges if x[1] == self.target_edge_type]
        src, _, dest = valid_edges[random.randint(0, len(valid_edges) - 1)]
        # a column that for each node specifies whether there is an edge to it from the source node:
        mask = np.ones(self.max_nodes, dtype=np.float)
        for e_type in range(self.edge_types):
            mask *= (1 - np.resize(a[:, e_type * self.max_nodes + src], self.max_nodes))
        mask[dest] = 0  # remove positive example from the mask - this line in redundant
        mask[n_nodes:] = 0
        return src, dest, mask

    def read_targets(self, graph):
        src = graph[self.targets][0]
        dest = graph[self.targets][1]
        mask = np.zeros(self.max_nodes, dtype=np.float)
        for node_id in graph[self.targets][2:]:
            mask[node_id] = 1
        return src, dest, mask