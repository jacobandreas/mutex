import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class EncDec(nn.Module):
    def __init__(self,
                 vocab_x,
                 vocab_y,
                 emb,
                 dim,
                 copy=False,
                 n_layers=1,
                 self_att=False,
                 dropout=0.,
                 bidirectional=True,
                 rnntype=nn.LSTM,
                 MAXLEN=45,
                ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.nll = nn.CrossEntropyLoss(ignore_index=vocab_y.pad(), reduction='mean') #TODO: why mean better?
        self.nll_wr = nn.CrossEntropyLoss(ignore_index=vocab_y.pad(), reduction='none')
        self.dim = dim
        self.n_layers = n_layers
        self.MAXLEN = MAXLEN

        if self.bidirectional:
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = nn.Identity()


        self.encoder = Encoder(vocab_x,
                               emb,
                               dim,
                               n_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               rnntype=rnntype)

        self.decoder = Decoder(vocab_y,
                               emb,
                               dim,
                               n_layers,
                               attention=None,
                               self_attention=self_att,
                               copy=copy,
                               dropout=dropout,
                               rnntype=rnntype,
                               concat_feed=False,
                               MAXLEN=self.MAXLEN,
                              )


    def pass_hiddens(self, rnnstate):
        if self.rnntype == nn.LSTM:
            state = [
                s.view(self.n_layers, -1, rnnstate[0].shape[1], self.dim).sum(dim=1)
                for s in rnnstate
            ]
        else:
            state = rnnstate.view(self.n_layers, -1, rnnstate.shape[1], self.dim).sum(dim=1)

        return state

    def forward(self, inp, out, per_instance=False):
        hid, state = self.encoder(inp)
        hid = self.proj(hid)
        state = self.pass_hiddens(state)
        out_src = out[:-1, :]

        dec, _, _, extras = self.decoder(state,
                                        out_src.shape[0],
                                        ref_tokens=out_src,
                                        att_features=None,
                                        att_tokens=None)

        if per_instance:
            out_tgt = out[1:, :].transpose(0,1)
            output = dec.permute(1,2,0)
            loss = self.nll_wr(output,out_tgt).sum(dim=-1)
        else:
            out_tgt = out[1:, :].view(-1)
            dec = dec.view(-1, len(self.vocab_y))
            loss = self.nll(dec, out_tgt)

        return loss

    def logprob(self, inp, out):
        hid, state = self.encoder(inp)
        hid = self.proj(hid)
        return self.decoder.logprob(out, rnn_state=self.pass_hiddens(state))

    def sample(
            self,
            inp,
            max_len,
            prompt=None,
            greedy=False,
            top_p=None,
            temp=1.0,
            custom_sampler=None,
            **kwargs):

        hid, state = self.encoder(inp)
        hid = self.proj(hid)
        state = self.pass_hiddens(state)

        return self.decoder.sample(state,
                                   max_len,
                                   temp=temp,
                                   greedy=greedy,
                                   top_p=top_p,
                                   custom_sampler=custom_sampler)

    def sample_with_gumbel(self, inp, max_len, temp=1.0, **kwargs):
        hid, state = self.encoder(inp)
        hid = self.proj(hid)
        state = self.pass_hiddens(state)
        return self.decoder.sample_with_gumbel(state, max_len, temp=temp)
