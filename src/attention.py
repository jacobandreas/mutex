import torch
from torch import nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=True, query=False, memory=False):
        super().__init__()
        self.key = key
        self.query = query
        self.memory = memory
        if self.key:
            self.make_key = nn.Linear(n_features, n_hidden)
        if self.query:
            self.make_query = nn.Linear(n_features, n_hidden)
        if self.memory:
            self.make_memory = nn.Linear(n_features, n_hidden)
        self.n_out = n_hidden

    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features
            
        if self.memory:
            memory = self.make_memory(features)
        else:
            memory = features
        
        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        # attention
        query = query.expand_as(key)
        scores = (key * query).sum(dim=2) 
        if mask is not None:
            scores += mask * -99999
        distribution = F.softmax(scores, dim=0)
        weighted = (memory * distribution.unsqueeze(2).expand_as(memory))
        summary = weighted.sum(dim=0, keepdim=True)

        # value
        return summary, distribution
