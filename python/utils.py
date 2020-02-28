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


def correct_func(anchor, positive, negative):
    pos_dist = tt.sum((anchor - positive) ** 2, 1)
    neg_dist = tt.sum((anchor - negative) ** 2, 1)

    return np.asscalar(tt.sum(pos_dist < neg_dist).detach().numpy())