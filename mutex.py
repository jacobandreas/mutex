import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import EncDec, Decoder, Vocab, batch_seqs, weight_top_p

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
                 rnntype=nn.LSTM):
        
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
        
        self.qxy = EncDec(vocab, 
                          emb, 
                          dim, 
                          copy=copy,
                          n_layers=n_layers,  
                          self_att=self_att, 
                          dropout=dropout, 
                          bidirectional=bidirectional,
                          rnntype=rnntype)
        
        self.px  = Decoder(vocab, 
                           emb, 
                           dim, 
                           copy=False, 
                           n_layers=n_layers, 
                           self_attention=False, 
                           dropout=dropout, 
                           rnntype=rnntype, 
                           concat_feed=False)


    def forward(self, inp, out):
        nll = self.pyx(inp, out)
        
        ys = self.py(self.Nsample, self.max_len)
        xs, _ = self.qxy.sample_with_gumbel(ys, self.max_len, temp=self.temp)
        logprob_pyx = -self.pyx(xs, ys, per_instance=True) # FIXME: Per instance loss might not be necessary here 
        
        with torch.no_grad():
            xps, logprob_px = self.px.sample(None, self.max_len, n_batch=self.Nsample)
            logprob_px = torch.Tensor(logprob_px).to(inp.device)
            xps =  batch_seqs(xps).to(inp.device)
            
        logprob_qxy = self.qxy.logprob(ys, xps)
        qxdpx = logprob_qxy-logprob_px
        point_kl   = torch.exp(qxdpx) * qxdpx
        return nll - self.lamda * (logprob_pyx.mean() - point_kl.mean())

    def sample_qxy(self, ys, temp=1.0):   
        tokens, _ = self.qxy.sample(ys, self.max_len, temp=temp)
        xlist = []
        for i in range(len(tokens)):
            xlist.append(self.vocab.decode(tokens[i]))
        return xlist
    
    def sample_px(self, batch, temp=1.0):       
        tokens, _ = self.px.sample(None, 
                                   self.max_len, 
                                   n_batch=batch, 
                                   temp=temp,
                                   att_features=None, 
                                   att_tokens=None, 
                                   greedy=False
                                   )
        xlist = []
        for i in range(len(tokens)):
            xlist.append(self.vocab.decode(tokens[i]))
        return xlist
    
    def sample_py(self, batch):     
        tokens  = self.py(batch, self.max_len).cpu().numpy()
        xlist = []
        for i in range(tokens.shape[1]):
            xlist.append(self.vocab.decode(tokens[:,i].flatten()))
        return xlist  

    def sample(self, *args, **kwargs):
        
        return self.pyx.sample(*args, **kwargs)