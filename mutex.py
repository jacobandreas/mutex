import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import EncDec, Decoder, Vocab, batch_seqs, weight_top_p
import random
from data import Oracle, collate
import collections
from itertools import combinations, product
import math

LossTrack = collections.namedtuple('LossTrack', 'nll mlogpyx pointkl')

class Mutex(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 emb,
                 dim,
                 py,
                 copy=False,
                 temp=1.0,
                 max_len_x=8,
                 max_len_y=8,
                 n_layers=1,
                 self_att=False,
                 dropout=0.,
                 bidirectional=True,
                 lamda=0.1,
                 Nsample=50,
                 rnntype=nn.LSTM,
                 px=None,
                 qxy=None,
                 kl_lamda=1.0,
                 regularize=True,
                 ent=0.0,
                ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.temp = temp
        self.lamda = lamda
        self.kl_lamda = kl_lamda
        self.ent = ent
        self.Nsample = Nsample
        self.MAXLEN_X = max_len_x
        self.MAXLEN_Y = max_len_y
        self.regularize=regularize
        self.py  = py

        self.pyx = EncDec(vocab_x,
                          vocab_y,
                          emb,
                          dim,
                          copy=copy,
                          n_layers=n_layers,
                          self_att=self_att,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          rnntype=rnntype,
                          MAXLEN=self.MAXLEN_Y)

        if qxy is None:
            self.qxy = EncDec(vocab_y,
                              vocab_x,
                              emb,
                              dim,
                              copy=copy,
                              n_layers=n_layers,
                              self_att=self_att,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              rnntype=rnntype,
                              MAXLEN=self.MAXLEN_X)
        else:
            self.qxy = qxy

        if px is None:
            self.px  = Decoder(vocab_x,
                               emb,
                               dim,
                               copy=False,
                               n_layers=n_layers,
                               self_attention=False,
                               dropout=dropout,
                               rnntype=rnntype,
                               concat_feed=False,
                               MAXLEN=self.MAXLEN_X)
        else:
            self.px = px

        self.loss_container = []

    def forward(self, inp, out):
        nll = self.pyx(inp, out)

        if not self.regularize: return nll

        ys  = self.py.sample(self.Nsample, self.MAXLEN_Y)

        # Hpy = math.log(len(self.py.vocab_y())) #

        # For Expactation term
        if isinstance(self.qxy, Oracle):
            xs = self.qxy.sample(ys, self.MAXLEN_X)
        else:
            xs, _ = self.qxy.sample_with_gumbel(ys, self.MAXLEN_X, temp=self.temp)

        logprob_pyx = -self.pyx(xs, ys)

        #For KL term, FIXME: currently we use enumerate to enumerate all xs
        if isinstance(self.qxy, Oracle):
            xps, _ = self.qxy.sample(self.Nsample, self.MAXLEN_X)
        else:
            with torch.no_grad():
                xs, _ = self.px.sample(None,
                                        self.MAXLEN_X,
                                        n_batch=self.Nsample,
                                        temp=self.temp,
                                        greedy=False)
                for _ in range(2):
                    qxs, _ = self.qxy.sample(ys, self.MAXLEN_X, temp=self.temp)
                    xs += qxs

                ux = [list(x) for x in set(tuple(x) for x in xs)] # UNIQUE XS
                #print(ux[0])
                xps   = batch_seqs(ux).to(inp.device)
                #xps  = batch_seqs(self.enumerate_xs()).to(inp.device)
                #print(xps.shape)
                #logprob_px = self.px.logprob(xps)

        #KL calculation
        point_kl, entropy = 0., 0.
        if not isinstance(self.qxy, Oracle):
            with torch.no_grad():
                logprob_px = self.px.logprob(xps)
            for y in ys.split(1,dim=1):
                ybatch = y.repeat(1, xps.shape[1])
                logprob_qxy = self.qxy.logprob(ybatch, xps)
                pqxy = torch.exp(logprob_qxy)
                point_kl += (pqxy * (logprob_qxy-logprob_px)).sum()
                entropy  += (pqxy * logprob_qxy).sum()
            point_kl = point_kl / self.Nsample
            entropy  = entropy /  self.Nsample

        self.loss_container.append(LossTrack(nll.item(), -logprob_pyx.item(), point_kl.item()))

        return nll - self.lamda * (logprob_pyx  - self.kl_lamda * point_kl) + self.ent * entropy 

    def sample_qxy(self, ys, temp=1.0):
        tokens, _ = self.qxy.sample(ys, self.MAXLEN_X, temp=temp)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().transpose().tolist()

        return self.print_tokens(self.vocab_x, tokens)

    def sample_qxy_gumbel(self, ys, temp=1.0):
        xs, _ = self.qxy.sample_with_gumbel(ys, self.MAXLEN_X, temp=self.temp)
        tokens = torch.argmax(xs,dim=-1)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().transpose().tolist()

        return self.print_tokens(self.vocab_x, tokens)

    def sample_px(self, batch, temp=1.0):
        if isinstance(self.px, Oracle):
            tokens, _ =  self.px.sample(batch, self.MAXLEN_X)
        else:
            tokens, _ = self.px.sample(None,
                                       self.MAXLEN_X,
                                       n_batch=batch,
                                       temp=temp,
                                       greedy=False
                                       )

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy().transpose().tolist()

        return self.print_tokens(self.vocab_x,tokens)


    def sample_py(self, batch):
        tokens  = self.py.sample(batch, self.MAXLEN_Y)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy().transpose().tolist()

        return self.print_tokens(self.vocab_y, tokens)

    def print_tokens(self, vocab, tokens):
        return [" ".join(vocab.decode(tokens[i]))
                    for i in range(len(tokens))]

    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)

    def enumerate_xs(self):
        data = []
        for i in range(1,self.MAXLEN_X):
            keywords = product(self.vocab_x._rev_contents.keys(), repeat = i)
            data = data + [ [self.vocab_x.sos()] + list(i)  for i in keywords]
        return data

    def sample_qxy_debug(self,N=10):
        with torch.no_grad():
            ys = self.py.sample(N,self.MAXLEN_Y, temp=self.temp)
            xs, _ = self.qxy.sample(ys, self.MAXLEN_X, temp=self.temp)
            xs   = batch_seqs(xs).to(ys.device)
            logprob_qxy = self.qxy.logprob(ys, xs).cpu().numpy().tolist()
            logprob_px = self.px.logprob(xs).cpu().numpy().tolist()
            logprob_pyx = -self.pyx(xs, ys, per_instance=True)
            xs  = xs.cpu().numpy().transpose().tolist()
            ys  = ys.cpu().numpy().transpose().tolist()
            sxs = self.print_tokens(self.vocab_x,xs)
            sys = self.print_tokens(self.vocab_y,ys)
            plist = []
            for (x,y,lqxy, lpx, lpyx) in zip(sxs, sys, logprob_qxy, logprob_px, logprob_pyx):
                plist.append(f"x: {x} \t y: {y} \t logpqxy: {lqxy} \t logpx: {lpx} \t logpyx {lpyx}")
            return plist

    def sample_qxy_debug_data(self, data):
        xs, ys = collate(data)
        device = self.px.embed.weight.device
        xs, ys = xs.to(device), ys.to(device)
        with torch.no_grad():
            logprob_qxy = self.qxy.logprob(ys, xs).cpu().numpy().tolist()
            logprob_px = self.px.logprob(xs).cpu().numpy().tolist()
            logprob_pyx = -self.pyx(xs, ys, per_instance=True)
            xs  = xs.cpu().numpy().transpose().tolist()
            ys  = ys.cpu().numpy().transpose().tolist()
            sxs = self.print_tokens(self.vocab_x, xs)
            sys = self.print_tokens(self.vocab_y, ys)
            plist = []
            for (x,y,lqxy, lpx, lpyx) in zip(sxs, sys, logprob_qxy, logprob_px, logprob_pyx):
                plist.append(f"x: {x} \t y: {y} \t logpqxy: {lqxy} \t logpx: {lpx} \t logpyx {lpyx}")
            return plist
