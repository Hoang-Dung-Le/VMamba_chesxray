# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch


# class SubsetRandomSampler(torch.utils.data.Sampler):
#     r"""Samples elements randomly from a given list of indices, without replacement.

#     Arguments:
#         indices (sequence): a sequence of indices
#     """

#     def __init__(self, indices):
#         self.epoch = 0
#         self.indices = indices

#     def __iter__(self):
#         return (self.indices[i] for i in torch.randperm(len(self.indices)))

#     def __len__(self):
#         return len(self.indices)

#     def set_epoch(self, epoch):
#         self.epoch = epoch


import torch

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Suitable for multi-label datasets.

    Arguments:
        indices (sequence): a sequence of indices
        labels (list of lists): a list of lists containing binary labels for each sample
    """

    def __init__(self, indices, labels):
        self.epoch = 0
        self.indices = indices
        self.labels = labels

    def __iter__(self):
        indices = torch.tensor(self.indices)  # Convert to tensor for indexing
        indices = indices[torch.randperm(len(indices))]  # Shuffle indices

        for index in indices:
            yield index.item(), self.labels[index]  # Yield index and its corresponding labels

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
