import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def batch_seqs(seqs):
    max_len = max(len(s) for s in seqs)
    data = np.zeros((max_len, len(seqs)))
    for i, s in enumerate(seqs):
            data[:len(s), i] = s
    return torch.LongTensor(data)

def weight_top_p(vec, p):
    indices = (-vec).argsort()
    out = np.zeros_like(vec)
    cumprob = 0
    for i in indices:
        excess = max(0, cumprob + vec[i] - p)
        weight = vec[i] - excess
        out[i] = weight
        cumprob += weight
        if excess > 0:
            break

    out /= out.sum()
    return out

def trim(L, obj):
    if obj in L:
        return L[:L.index(obj)+1]
    return L