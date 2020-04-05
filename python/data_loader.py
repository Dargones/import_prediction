# A module for loading the graph data from .json files and preparing them for input to GGNNs

import numpy as np
from torch.utils.data import Dataset
import random


class GraphDataset(Dataset):
    """
    This class allows converting graphs to their neural-network representations on demand
    """

    def __init__(self, data, hidden_size, max_nodes, edge_types, annotation_size, targets, target_edge_type, max_targets=10):
        """
        Initialize GraphDataset so that it can be passed to DataLoader
        :param data:            graph data in .json format as loaded from disk
        :param hidden_size:     the size of node embedding in GGNN that will be used on this dataset
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

    def get_numeric_representation(self, index):
        """
        Given an index of a graph in the dataset, get its numeric representation. This consists
        of the following:
        1) adjacency matrix
        2) matrix of initial node annotations
        3) src, pos, and mask as returned by create_target() or read_target()
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

    def __getitem__(self, index):
        """
        For a given repository, return its numeric representation with the target edge removed
        as well as max_targets more of its numeric representations each of which introduces a
        single edge to the graph. The network will have to select the graph that introduces the
        target edge
        """
        matrix, features, mask, src, pos = self.get_numeric_representation(index)

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
        a = np.zeros([self.max_nodes, self.max_nodes * self.edge_types * 2])
        for edge in edges:
            src = edge[0]
            e_type = edge[1]
            dest = edge[2]
            self.set_matrix(a, src, dest, e_type, 1)
        return a

    def set_matrix(self, a, src, dest, e_type, value):
        """
        Remove or add an edge in the adjacency matrix. Also remove or add the corresponding edge
        going in the opposite direction
        :param a:     the adjacency matrix
        :param src:   the source node
        :param dest:  the destination node
        :param e_type:the type of the edge to be removed\added
        :param value: 1 if the edge is to be added, 0 otehrwise
        """
        a[dest][(e_type - 1) * self.max_nodes + src] = value
        a[src][(e_type - 1 + self.edge_types) * self.max_nodes + dest] = value

    def create_target(self, edges, a, n_nodes):
        """
        Select a random import edge on a graph. Return the src and destination nodes that this
        edge connects. Also return a mask that for each node in the graph specifies whether it
        could have been connected to the src node with an import edge. These are future negatives
        that the network will have to distinguish from the true destination node.
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
        """
        Read the targets directly from the json file (should be used for testing). Return the
        targets in the same format that create target does
        """
        src = graph[self.targets][0]
        dest = graph[self.targets][1]
        mask = np.zeros(self.max_nodes, dtype=np.float)
        for node_id in graph[self.targets][2:]:
            mask[node_id] = 1
        return src, dest, mask