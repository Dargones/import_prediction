import numpy as np
import torch as tt
from torch.utils.data import Dataset, DataLoader
import random
import os
import json

class GraphDataLoader:

    def __init__(self, directory, hidden_size, directed=False, max_nodes=None, edge_types=None, annotation_size=None):
        self.directory = directory
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size

    def load(self, filename, batch_size, new_targets=False, shuffle=False, retain_string=False):
        """
        Method that loads the data in memory.
        Note that targets can be absolutely anything at this point
        :param filename:       name of the .json file with graphs
        :param create_targets: if True, create targets automatically on each pass
        :return:
        """
        full_path = os.path.join(self.directory, filename) # full path to the file
        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        self.update_parameters(data)
        dataset =  GraphDataset(data, self.hidden_size, self.directed, self.max_nodes,
                            self.edge_types, self.annotation_size, new_targets, retain_string)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    def update_parameters(self, data):
        """
        Given a new dataset, update such parameters as the maximum number of nodes and edge
        types if these parameters have not been set in advance. If the parameters have already
        been initialized, assert that the parameters match
        :param data:
        :return:
        """
        edge_types_tmp = 0
        max_nodes_tmp = 0
        annotation_size_tmp = len(data[0]["annotations"][0])

        for graph in data:
            max_nodes_tmp = max(max_nodes_tmp, len(graph["annotations"]))
            edge_types_tmp = max(edge_types_tmp, max([e[1] for e in graph['edges']]))

        if not self.directed:
            edge_types_tmp *= 2
        if not self.edge_types:
            self.edge_types = edge_types_tmp
        if not self.max_nodes:
            self.max_nodes = max_nodes_tmp
        if not self.annotation_size:
            self.annotation_size = annotation_size_tmp

        assert(edge_types_tmp <= self.edge_types)
        assert(max_nodes_tmp <= self.max_nodes)
        assert(annotation_size_tmp == annotation_size_tmp)


class GraphDataset(Dataset):
    """
    This class allows converting graphs to their neural-network representations on demand
    """

    def __init__(self, data, hidden_size, directed, max_nodes, edge_types, annotation_size, new_targets, retain_strings):
        self.data = data
        self.hidden_size = hidden_size
        self.directed = directed
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.annotation_size = annotation_size
        self.new_targets = new_targets
        self.retain_strings = retain_strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        graph = self.data[index]
        adj_matrix = self.create_adjacency_matrix(index)

        features = np.pad(graph['annotations'],
                                 ((0, self.max_nodes - len(graph['annotations'])),
                                  (0, self.hidden_size - self.annotation_size)),  # left-right
                                 'constant')

        if self.new_targets:
            targets = self.create_target(graph['edges'], adj_matrix)
        else:
            targets = self.extract_target(graph['targets'])

        if not self.retain_strings:
            return tt.FloatTensor(adj_matrix), tt.FloatTensor(features), tt.LongTensor(targets)

        return tt.FloatTensor(adj_matrix), \
               tt.FloatTensor(features), \
               tt.LongTensor(targets),\
               graph["strings"]


    def create_adjacency_matrix(self, index):
        """
        Create adjacency matrix for the graph that is indexed as index in the dataset
        :param index:
        :return:
        """
        a = np.zeros([self.max_nodes, self.max_nodes * self.edge_types])
        for edge in self.data[index]["edges"]:
            src = edge[0]
            e_type = edge[1]
            dest = edge[2]
            a[dest - 1][(e_type - 1) * self.max_nodes + src - 1] = 1  # a[target][source]
            if self.directed:
                continue
            a[src - 1][(e_type - 1 + int(self.edge_types/2)) * self.max_nodes + dest - 1] = 1
        return a

    def create_target(self, edges, a):
        """
        Modify the graph by removing an edge. Return a triplet of node indices that are to be used
        later during loss finction calculation
        :return:
        """
        src, e_type, dest = edges[random.randint(0, len(edges) - 1)]  # select a random edge
        # a column that for each node specifies whether there is an edge to it from the source node:
        src_col = np.resize(a[:, (e_type - 1) * self.max_nodes + src - 1], self.max_nodes)
        # list of nodes to which there is no edge from source:
        src_col_edges = np.where(src_col==0)[0]

        if len(src_col_edges) == 0:
            print("All edges there")
            fake_dest = ((dest + 1) % self.max_nodes) + 1
        else:
            fake_dest = src_col_edges[random.randint(0, len(src_col_edges) - 1)] + 1

        a[dest - 1][(e_type - 1) * self.max_nodes + src - 1] = 0
        a[fake_dest - 1][(e_type - 1) * self.max_nodes + src - 1] = 1

        if not self.directed:
            a[src - 1][(e_type - 1 + int(self.edge_types / 2)) * self.max_nodes + dest - 1] = 0
            a[src - 1][(e_type - 1 + int(self.edge_types / 2)) * self.max_nodes + fake_dest - 1] = 1

        return [[src - 1] * self.hidden_size,
                [dest - 1] * self.hidden_size,
                [fake_dest - 1] * self.hidden_size]

    def extract_target(self, target):
        return None