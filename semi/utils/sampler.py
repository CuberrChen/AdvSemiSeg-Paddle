import math
import numpy as np
import paddle
from paddle.io import Sampler

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in paddle.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


__all__ = ["SubsetBatchSampler"]


class SubsetBatchSampler(Sampler):

    def __init__(self,
                 indices=None,
                 shuffle=True,
                 batch_size=1,
                 drop_last=False):
        self.indices = indices
        if shuffle:
            self.sampler = SubsetRandomSampler(self.indices)

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer, but got {}".format(batch_size)
        self.batch_size = batch_size
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean value, but got {}".format(type(drop_last))
        self.drop_last = drop_last

    def __iter__(self):
        batch_indices = []
        for idx in self.sampler:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = len(self.sampler)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size