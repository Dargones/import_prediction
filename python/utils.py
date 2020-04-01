#!/usr/bin/env/python

# This module contains helper classes that are used elsewhere throughout the code

import torch as tt
import numpy as np
from torch.nn.modules.loss import _Loss
import queue
import threading


class ThreadedIterator:
    """
    The class is copied over from Microsoft Research Github:
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/utils.py
    An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None
    """

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class LossLessTripletLoss(_Loss):
    """
    Pytorch implementation of the "lossless" triplet loss. The rationale behind this version of
    the loss can be found here: https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    The original function was written for Tensorflow/Keras and the code can be found here:
    https://gist.github.com/marcolivierarsenault/a7ef5ab45e1fbb37fbe13b37a0de0257
    """

    def __init__(self, dim, size_average=None, reduce=None, reduction='mean', e=1e-8):
        super(LossLessTripletLoss, self).__init__(size_average, reduce, reduction)
        self.dim = dim
        self.e = e

    def forward(self, anchor, positive, negative):
        pos_dist = tt.sum((anchor - positive) ** 2, 1)
        neg_dist = tt.sum((anchor - negative) ** 2, 1)

        pos_dist = -tt.log(-pos_dist / self.dim + 1 + self.e)
        neg_dist = -tt.log(-(self.dim - neg_dist) / self.dim + 1 + self.e)

        return tt.sum(neg_dist + pos_dist)


class CumulativeTripletLoss(_Loss):

    def __init__(self, dim, size_average=None, reduce=None, reduction='mean', e=1e-8):
        super(CumulativeTripletLoss, self).__init__(size_average, reduce, reduction)
        self.dim = dim
        self.e = e

    def forward(self, embeds, mask, src, pos, oneNegAcc=True):
        """

        :param embeds: (BATCH, MAX_NODES, HIDDEN)
        :param mask:   (BATCH, MAX_NODES)
        :param src:    (BATCH)
        :param pos:    (BATCH)
        :return:
        """
        # embeddings corresponding to the source node (BATCH, HIDDEN):
        src_embeds = tt.gather(embeds, 1, src.view(-1, 1).unsqueeze(2).repeat(1, 1, embeds.shape[2]))
        # distances between source and other nodes (BATCH, MAX_NODES):
        all_dist = tt.sum((embeds - src_embeds) ** 2, 2)
        # distance between source and positive (BATCH, 1):
        pos_dist = tt.gather(all_dist, 1, pos.view(-1, 1))
        pos_dist_log = -tt.log(-pos_dist / self.dim + 1 + self.e)
        # distance between source and all negatives (BATCH, MAX_NODES):
        neg_dist = all_dist * mask
        neg_dist_log = -tt.log(-(self.dim - all_dist) / self.dim + 1 + self.e)
        neg_dist_log = neg_dist_log * mask
        # Total negative for each batch (BATCH):
        neg_total = tt.sum(mask > 0, 1).type(tt.float64)
        # result
        loss = tt.sum(pos_dist_log.view(-1) + tt.sum(neg_dist_log, 1).type(tt.float64) / neg_total)
        acc = tt.sum(neg_dist> pos_dist, 1).type(tt.float64)/neg_total
        return loss, tt.mean(acc).detach().numpy().item()


class SimilarityLoss(_Loss):

    def __init__(self, margin=0.5, size_average=None, reduce=None, reduction='mean'):
        super(SimilarityLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, diff, mask):
        """
        :return:
        """
        negative_total = tt.sum(diff * mask, dim = 1)
        negative = negative_total / tt.sum(mask, dim=1)
        positive = diff[:, 1]
        loss = positive - negative + self.margin

        positive_repeated = positive.view(-1, 1).repeat((1, diff.shape[1]))
        acc = tt.sum((positive_repeated < diff * mask).double() *mask, dim=1)/tt.sum(mask, dim=1)

        return tt.mean(loss), tt.mean(acc)





def correct_func(anchor, positive, negative):
    pos_dist = tt.sum((anchor - positive) ** 2, 1)
    neg_dist = tt.sum((anchor - negative) ** 2, 1)

    return np.asscalar(tt.sum(pos_dist < neg_dist).detach().numpy())