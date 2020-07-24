import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=True,
            dropout=0,
            rnntype=nn.LSTM,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = rnntype(
            n_embed, n_hidden, n_layers, bidirectional=bidirectional
        )

    def forward(self, data):
        if len(data.shape) == 3:
            emb    = torch.matmul(data, self.embed.weight)
            tokens = torch.argmax(data.detach(),dim=-1)  
            emb    = emb * (tokens != self.vocab.pad()).unsqueeze(2).float()
        else:
            emb   = self.embed(data)
        
        return self.rnn(self.embed_dropout(emb))
    
