import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import EncDec, Decoder, Vocab, batch_seqs, weight_top_p
import random
from data import Oracle
import collections
from itertools import combinations, product

LossTrack = collections.namedtuple('LossTrack', 'nll mlogpyx pointkl')

class Mutex(nn.Module):
    def __init__(self, 
                 vocab, 
                 emb, 
                 dim, 
                 py, 
                 copy=False, 
                 temp=1.0, 
                 max_len=8, 
                 n_layers=1, 
                 self_att=False, 
                 dropout=0., 
                 bidirectional=True,
                 lamda=0.1,
                 Nsample=50,
                 rnntype=nn.LSTM,
                 px=None,
                 qxy=None
                ):
        
        super().__init__()
                
        self.vocab = vocab
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.max_len = max_len
        self.temp = temp
        self.lamda = lamda
        self.Nsample = Nsample
        
        self.py  = py
        
        self.pyx = EncDec(vocab, 
                          emb, 
                          dim, 
                          copy=copy, 
                          n_layers=n_layers,  
                          self_att=self_att, 
                          dropout=dropout, 
                          bidirectional=bidirectional, 
                          rnntype=rnntype)
        
        if qxy is None:
            self.qxy = EncDec(vocab, 
                          emb, 
                          dim, 
                          copy=copy,
                          n_layers=n_layers,  
                          self_att=self_att, 
                          dropout=dropout, 
                          bidirectional=bidirectional,
                          rnntype=rnntype)
        else:
            self.qxy = qxy
            
        if px is None:   
            self.px  = Decoder(vocab, 
                           emb, 
                           dim, 
                           copy=False, 
                           n_layers=n_layers, 
                           self_attention=False, 
                           dropout=dropout, 
                           rnntype=rnntype, 
                           concat_feed=False)
        else:
            self.px = px

        self.loss_container = []

    def forward(self, inp, out):
        nll = self.pyx(inp, out)
        
        ys = self.py.sample(self.Nsample, self.max_len)
        
        if isinstance(self.qxy, Oracle):
            xs = self.qxy.sample(ys, self.max_len)
        else:
            with torch.no_grad():
                xs, _ = self.qxy.sample_with_gumbel(ys, self.max_len, temp=self.temp)
        
        logprob_pyx = -self.pyx(xs, ys, per_instance=False)
        
        if isinstance(self.qxy, Oracle):
            xps, _ = self.qxy.sample(self.Nsample, self.max_len)
        else:
            with torch.no_grad():     
            #xs, _ = self.qxy.sample(ys, self.max_len, temp=self.temp)
#                 ux    = np.unique(np.array(xs), axis=0).tolist()
                xs = self.enumerate_xs()
                xps   = batch_seqs(xs).to(inp.device)
#                 print(xps.shape)
#                 logprob_px = self.px.logprob(xps)
        
        if isinstance(self.qxy, Oracle):
            point_kl = 0.0
        else:
            point_kl = 0
            with torch.no_grad():   
                logprob_px = self.px.logprob(xps)
            for y in ys.split(1,dim=1):
                ybatch = y.repeat(1, xps.shape[1])
                logprob_qxy = self.qxy.logprob(ybatch, xps)
                qxdpx = logprob_qxy-logprob_px #include all xs in a 
                point_kl += (torch.exp(logprob_qxy) * qxdpx).sum()
                
            point_kl = point_kl/self.Nsample
    
        self.loss_container.append(LossTrack(nll.item(), -logprob_pyx.item(), point_kl.item()))
        
        return nll - self.lamda * (logprob_pyx - point_kl)

    def sample_qxy(self, ys, temp=1.0):
        tokens, _ = self.qxy.sample(ys, self.max_len, temp=temp)        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().transpose().tolist()

        return self.print_tokens(tokens)
    
    def sample_px(self, batch, temp=1.0):   
        if isinstance(self.px, Oracle):
            tokens, _ =  self.px.sample(batch, self.max_len)
        else:
            tokens, _ = self.px.sample(None, 
                                       self.max_len, 
                                       n_batch=batch, 
                                       temp=temp,
                                       greedy=False
                                       )
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy().transpose().tolist()
            
        return self.print_tokens(tokens)
    
    
    def sample_py(self, batch):     
        tokens  = self.py.sample(batch, self.max_len)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().transpose().tolist()        
            
        return self.print_tokens(tokens)

    def print_tokens(self, tokens):     
        return [" ".join(self.vocab.decode(tokens[i]))
                    for i in range(len(tokens))]
    
    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)
    
    def enumerate_xs(self):
        data = []
        for i in range(1,self.max_len):
            keywords = product(self.vocab._rev_contents.keys(), repeat = i)
            data = data + [ [self.vocab.sos()] + list(i)  for i in keywords]
        return data
                