# utils.py di folder GraphSAINT/graphsaint/pytorch_version
import numpy as np
import torch
from torch.autograd import Variable


def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()
