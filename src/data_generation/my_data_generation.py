
import numpy as np
import torch
from torch import Tensor
from util.misc import NestedTensor

class DataGenerator:
    def __init__(self, params):
        self.params = params
        self.device = params.training.device

    def get_batch(self,training_data,labels):

        training_data = tuple(training_data)

        labels = tuple(labels)
        labels = [Tensor(l).to(torch.device(self.device)) for l in labels]

        # Pad training data
        max_len = max(list(map(len, training_data)))
        training_data, mask = pad_to_batch_max(training_data, max_len)

        training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
        Tensor(mask).bool().to(torch.device(self.device)))

        return training_nested_tensor, labels


def pad_to_batch_max(training_data, max_len):
    batch_size = len(training_data)
    d_meas = training_data[0].shape[1]
    training_data_padded = np.zeros((batch_size, max_len, d_meas))
    mask = np.ones((batch_size, max_len))
    for i, ex in enumerate(training_data):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0

    return training_data_padded, mask
