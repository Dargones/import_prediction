#!/usr/bin/env/python
# This module contains helper classes that are used elsewhere throughout the code

import torch as tt
from torch.nn.modules.loss import _Loss


class CustomTripletLoss(_Loss):
    """
    This class provides functionality similar to that of the standard triplet loss, except that
    it gets several negatives as input, each of which is compared separately to the positive. The
    resulting loss value is the mean across the negatives
    """

    def __init__(self, margin=0.5, size_average=None, reduce=None, reduction='mean'):
        super(CustomTripletLoss, self).__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, diff, mask):
        """
        :diff:    # (BATCH_SIZE, N_OF_NEGATIVES + 2) - the distance from the anchor to
                    itself (diff[0]), the positive(diff[1]), and the negatives [diff[2:]]
        :mask:    # (BATCH_SIZE, N_OF_NEGATIVES + 2) - because each datapoint in a batch can have
                    a different number of negatives, one has to use the mask. The mask has zeroes
                    at positions corresponding to negatives taht should not be taken into account
        :return:  loss, and accuracy (probability of positive being closer to anchor than negative)
        """
        positive = diff[:, 1].view(-1, 1).repeat((1, diff.shape[1]))
        loss = tt.nn.functional.relu(positive - diff + self.margin) * mask
        loss = tt.sum(loss, dim=1)/tt.sum(mask, dim=1)
        acc = tt.sum((positive < diff * mask).double() *mask, dim=1)/tt.sum(mask, dim=1)
        return tt.mean(loss), tt.mean(acc)