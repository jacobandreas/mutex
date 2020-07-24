from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .attention import SimpleAttention
from .utils import weight_top_p

#_VF = torch._C._VariableFunctions
EPS = 1e-7

DecoderState = namedtuple("DecoderState", "feed rnn_state hiddens tokens")
BeamState = namedtuple("BeamState", "feed rnn_state hiddens tokens score parent done")

class Decoder(nn.Module):
    def __init__(
            self, 
            vocab, 
            n_embed, 
            n_hidden, 
            n_layers, 
            attention=None,
            copy=False,
            self_attention=False,
            dropout=0,
            rnntype=nn.LSTM,
            concat_feed=True,
    ):
        super().__init__()

        # setup
        self.vocab = vocab
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.copy = copy
        self.self_attention = self_attention
        self.rnntype=rnntype
        # attention
        if attention is None:
            attention = ()
        attention = tuple(attention)
        if self_attention:
            attention = attention + (SimpleAttention(n_hidden, n_hidden),)
        self.attention = attention
        for i, att in enumerate(attention):
            self.add_module("attention_%d" % i, att)
        # modules
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.combine = nn.Linear(n_hidden * (1 + len(attention)), n_hidden)
        self.dropout_in = nn.Dropout(dropout)
        self.predict = nn.Linear(n_hidden, len(vocab))
        #self.copy_switch = nn.Linear(n_hidden, 1 + len(attention))
        self.concat_feed = concat_feed
        if self.concat_feed:
            self.rnn = self.rnntype(n_embed + n_hidden, n_hidden, n_layers)
        else:
            self.rnn = self.rnntype(n_embed, n_hidden, n_layers)
        self.dropout_out = nn.Dropout(dropout)
        self.seq_picker = nn.Linear(n_hidden, len(attention))
        self.nll = nn.CrossEntropyLoss(ignore_index=vocab.pad(), reduction='none')
        self.nllreduce = nn.CrossEntropyLoss(ignore_index=vocab.pad())
        
            
    def step(
            self,
            decoder_state,
            att_features,
            att_tokens,
            att_masks,
            att_token_proj,
            self_att_proj
    ):
        # advance rnn
        if decoder_state.tokens.dtype == torch.float32:
            emb = torch.matmul(decoder_state.tokens, self.embed.weight)
        else:
            emb = self.embed(decoder_state.tokens[-1, :])
            
        if self.concat_feed:
            inp = self.dropout_in(torch.cat((emb, decoder_state.feed), dim=1))
        else:
            inp = self.dropout_in(emb)
        

        hidden, rnn_state = self.rnn(inp.unsqueeze(0), decoder_state.rnn_state)
        hiddens = torch.cat(decoder_state.hiddens + [hidden], dim=0)

        # prep self-attention
        if self.self_attention:
            att_features = tuple(att_features) + (hiddens,)
            att_tokens = tuple(att_tokens) + (decoder_state.tokens,)
            att_masks = att_masks + (
                (decoder_state.tokens == self.vocab.pad()).float(),
            )
            att_token_proj = att_token_proj + (self_att_proj,)

        # advance attention
        attended = [
            attention(features, hidden, mask) 
            for attention, features, mask in zip(
                self.attention, att_features, att_masks
            )
        ]
        if len(attended) > 0:
            summary, distribution = zip(*attended)
        else:
            summary = distribution = ()
            
            
        all_features = torch.cat([hidden] + list(summary), dim=2)
        comb_features = self.dropout_out(self.combine(all_features).squeeze(0))
        pred_logits = self.predict(comb_features)
        
        assert not torch.isnan(pred_logits).any()
        
        # copy mechanism
        ### if self.copy:
        ###     pred_probs = F.softmax(pred_logits, dim=1)
        ###     copy_probs = [
        ###         (dist.unsqueeze(2) * proj).sum(dim=0)
        ###         for dist, proj in zip(distribution, att_token_proj)
        ###     ]
        ###     all_probs = torch.stack([pred_probs] + copy_probs, dim=1)
        ###     copy_weights = F.softmax(self.copy_switch(comb_features), dim=1)
        ###     comb_probs = (copy_weights.unsqueeze(2) * all_probs).sum(dim=1)
        ###     pred_logits = torch.log(comb_probs)
        if self.copy:
            pred_probs = F.softmax(pred_logits, dim=1)
            dists = distribution
            projs = att_token_proj

            seq_probs = self.seq_picker(hidden).softmax(dim=2)
            copy_weights = pred_probs[:, self.vocab.copy()].unsqueeze(0)
            #print(seq_probs[0, 0, :], copy_weights[0, 0])
            weighted_dists = [dists[i] * seq_probs[:, :, i] * copy_weights for i in range(len(dists))]
            copy_probs = [(weighted_dists[i].unsqueeze(2) *
                projs[i]).sum(0) for i in range(len(dists))]
            copy_probs = sum(copy_probs)

            copy_probs += EPS
            comb_probs = (
                copy_probs + pred_probs
            )
            comb_probs[:, self.vocab.copy()] = 0
            direct_logits = pred_logits
            copy_logits = torch.log(copy_probs)
            pred_logits = torch.log(comb_probs)
        else:
            direct_logits = pred_logits
            copy_logits = None
            weighted_dists = None

        # done
        return (
            pred_logits,
            comb_features,
            rnn_state,
            hidden,
            direct_logits, copy_logits, weighted_dists
        )

    def _make_projection(self, tokens):
        proj = tokens.new_zeros(
            tokens.shape[0], tokens.shape[1], len(self.vocab)
        ).float()
        for i in range(tokens.shape[0]):
            proj[i, range(tokens.shape[1]), tokens[i, :]] = 1
            #proj[i, :, tokens[i, :]] = 1
        return proj

    def forward(
            self,
            rnn_state,
            max_len,
            ref_tokens=None,
            att_features=None,
            att_tokens=None,
            token_picker=None
    ):

        # token picker
        if token_picker is None:
            master_self_att_proj = self._make_projection(ref_tokens)
            def token_picker(t, logits):
                return ref_tokens[t, :], master_self_att_proj[:t+1, :, :]

        # attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            assert len(att_features) == len(att_tokens)
            assert len(self.attention) == len(att_features) + (1 if self.self_attention else 0)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # init
        pred = None
        dummy_tokens, _ = token_picker(0, pred)
        feed = dummy_tokens.new_zeros(
            dummy_tokens.shape[0], self.n_hidden
        ).float()
        hiddens = []
        all_tokens = []
        all_preds = []
        all_extra = []

        # iter
        for t in range(max_len):
            tokens, self_att_proj = token_picker(t, pred)
            if tokens is None:
                break
            all_tokens.append(tokens)
            decoder_state = DecoderState(
                feed, rnn_state, hiddens, torch.stack(all_tokens)
            )

            pred, feed, rnn_state, hidden, *extra = self.step(
                decoder_state,
                att_features,
                att_tokens,
                att_masks,
                att_token_proj,
                self_att_proj,
            )
            hiddens.append(hidden)
            all_preds.append(pred)
            all_extra.append(extra)

        return (
            torch.stack(all_preds),
            torch.stack(all_tokens),
            rnn_state,
            list(zip(*all_extra))
        )

    def sample(
            self,
            rnn_state,
            max_len,
            n_batch=1,
            temp=1.0,
            att_features=None,
            att_tokens=None,
            greedy=False,
            top_p=None,
            custom_sampler=None,
           
    ):
        
        device = self.embed.weight.device
        
        # init
        if self.rnntype == nn.LSTM:
            if rnn_state is None:
                rnnstate = tuple(torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device) for _ in range(2))
            else:
                n_batch = rnn_state[0].shape[1]
        else:
            if rnn_state is None:
                rnnstate = torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device)
            else:
                n_batch = rnn_state.shape[1] 

        done = [False for _ in range(n_batch)]
        running_proj = torch.zeros(max_len, n_batch, len(self.vocab)).to(device)
        running_out = torch.zeros(max_len, n_batch, dtype=torch.int64).to(device)

        def token_picker(t, logits):
            # first step
            if t == 0:
                toks = torch.LongTensor(
                    [self.vocab.sos() for _ in range(n_batch)]
                ).to(device)
                running_proj[0, range(n_batch), toks] = 1
                running_out[0, range(n_batch)] = 1
                return toks, running_proj[:1, :, :]

            if all(done):
                return None, None

            # sample
            probs = F.softmax(logits/temp, dim=1)
            probs = probs.detach().cpu().numpy()
            tokens = []
            for i, row in enumerate(probs):
                if done[i]:
                    tokens.append(self.vocab.pad())
                    continue          
                if self.copy:
                    row[self.vocab.copy()] = 0
                    
                if custom_sampler is not None:
                    row = torch.tensor(row).unsqueeze(0)
                    choice = custom_sampler(row, running_out[:t])
                elif greedy:
                    choice = np.argmax(row)
                elif top_p:
                    row = weight_top_p(row, top_p)
                    choice = np.random.choice(len(self.vocab), p=row)
                else:
                    row /= row.sum()
                    choice = np.random.choice(len(self.vocab), p=row)
                tokens.append(choice)
                if choice == self.vocab.eos():
                    done[i] = True

            toks = torch.LongTensor(tokens).to(device)
            running_proj[t, range(n_batch), toks] = 1
            running_out[t, range(n_batch)] = toks
            return toks, running_proj[:t+1, :, :]

        preds, tokens, rnn_state, *_ = self(
            rnn_state,
            max_len,
            att_features=att_features,
            att_tokens=att_tokens,
            token_picker=token_picker
        )
        tok_arr = tokens.detach().cpu().numpy().transpose()
        tok_out = []
        score_out = [0 for _ in range(tok_arr.shape[0])]
        for i, row in enumerate(tok_arr):
            row_out = []
            for t, c in enumerate(row):
                row_out.append(int(c))
                score_out[i] += preds[t, i, c].item()
                if c == self.vocab.eos():
                    break
            tok_out.append(row_out)
        return tok_out, score_out

    def sample_with_gumbel(self,
                           rnn_state,
                           max_len,
                           n_batch=1,
                           att_features=None,
                           att_tokens=None,
                           temp=1.0):
         
        device = self.embed.weight.device
        # init
        if self.rnntype == nn.LSTM:
            if rnn_state == None:
                rnnstate = tuple(torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device) for _ in range(2))
            else:
                n_batch = rnn_state[0].shape[1]
        else:
            if rnn_state == None:
                rnnstate = torch.zeros(self.n_layers, n_batch, self.n_hidden).to(device)
            else:
                n_batch = rnn_state.shape[1] 

        done = [False for _ in range(n_batch)]
        running_proj = torch.zeros(max_len, n_batch, len(self.vocab)).to(device)
        running_out = torch.zeros(max_len, n_batch, dtype=torch.int64).to(device)
        def token_picker(t, logits):
            # first step
            if t == 0:
                toks = torch.LongTensor(
                    [self.vocab.sos() for _ in range(n_batch)]
                ).to(device)
                onehots = F.one_hot(toks, num_classes=len(self.vocab)).float()
                running_proj[0, range(n_batch), toks] = 1
                running_out[0, range(n_batch)] = self.vocab.sos()
                return toks, running_proj[:1, :, :], onehots

            if all(done):
                return None, None, None

            # sample
            onehots = F.gumbel_softmax(logits, tau=temp, hard=True, dim=1)
            toks = []
            for i in range(onehots.shape[0]):
                choice = torch.argmax(onehots[i].detach()).item()
                
                if self.copy:
                    onehots[:,self.vocab.copy()] = 0
                    
                if done[i]:
                    onehots[i,choice] = 0
                    onehots[i,self.vocab.pad()] = 1
                    toks.append(self.vocab.pad())
                    continue 
                    
                toks.append(choice)
                
                if choice == self.vocab.eos():
                    done[i] = True
                    
                
            toks = torch.LongTensor(toks).to(device)
            running_proj[t, range(n_batch), toks] = 1
            running_out[t, range(n_batch)] = toks
            return toks, running_proj[:t+1, :, :], onehots

        preds, tokens, rnn_state, *_ = self.forward_onehot(
            rnn_state,
            max_len,
            att_features=att_features,
            att_tokens=att_tokens,
            token_picker=token_picker
        )
        #tgt_tokens: (seq_len, batch_size)
        tgt_tokens = torch.argmax(tokens.detach(),dim=-1)[1:]
        # preds: (seq_len,batch_size,vocab_size)
        # output:  (batch_size,vocab_size,seq_len)
        output = preds[:-1,:,:].transpose(0,1).transpose(1,2)
        
        logprobs = -self.nll(output,tgt_tokens.transpose(0,1)).sum(dim=-1)
        
        return tokens, logprobs
    
    def forward_onehot(self,
            rnn_state,
            max_len,
            ref_tokens=None,
            att_features=None,
            att_tokens=None,
            token_picker=None,
            ):
        
        if token_picker is None:
            master_self_att_proj = None
            #token_picker = lambda t, logits: (
            #    (ref_tokens[t, :], self_att_proj[:t+1, :, :])
            #)
            def token_picker(t, logits):
                onehots = ref_tokens[t, :, :]
                tokens = torch.argmax(ref_tokens[t, :, :].detach(),dim=1)
                return tokens, None, onehots
        # attention
        if att_features is None:
            att_features = ()
            att_tokens = () 
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            assert len(att_features) == len(att_tokens)
            assert len(self.attention) == len(att_features) + (1 if self.self_attention else 0)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # init
        pred = None
        dummy_tokens, _ , dummy_onehots = token_picker(0, pred)
        feed = dummy_tokens.new_zeros(
            dummy_tokens.shape[0], self.n_hidden
        ).float()
        hiddens = []
        all_tokens = []
        all_tokens_onehot = []
        all_preds = []
        all_extra = []

        # iter
        for t in range(max_len):
            tokens, self_att_proj, onehots = token_picker(t, pred)
            if tokens is None:
                break
            all_tokens.append(tokens)
            all_tokens_onehot.append(onehots)
            
            decoder_state = DecoderState(
                feed, rnn_state, hiddens, onehots
            )

            pred, feed, rnn_state, hidden, *extra = self.step(
                decoder_state,
                att_features,
                att_tokens,
                att_masks,
                att_token_proj,
                self_att_proj,
            )
            hiddens.append(hidden)
            all_preds.append(pred)
            all_extra.append(extra)
        return (
            torch.stack(all_preds),
            torch.stack(all_tokens_onehot, dim=0),
            rnn_state,
            list(zip(*all_extra))
        )
    
    def logprob(self, ref_tokens, rnn_state=None):
        if len(ref_tokens.shape) == 3:
            emb = torch.matmul(ref_tokens[:-1,:,:], self.embed.weight)
            ids = torch.argmax(ref_tokens.detach(),dim=-1)
            in_mask = (ids[:-1,:] != self.vocab.pad()).unsqueeze(2).float()
            pred, _  = self.rnn(emb * in_mask, rnn_state)
            outmask = (ids[1:,:] != self.vocab.pad()).unsqueeze(2).float()
            logp = F.log_softmax(pred,dim=-1)
            logprob = (logp * (ref_tokens[1:,:,:] * outmask)).sum(dim=-1).sum(dim=0)       
        else:
            emb = self.embed(ref_tokens[:-1,:])
            tgt_tokens = ref_tokens[1:,:].transpose(0,1)  
            pred, _ = self.rnn(emb, rnn_state)
            output = pred.transpose(0,1).transpose(1,2) # TODO: too much transpose is going here
            logprob = -self.nll(output,tgt_tokens).sum(dim=-1)
        return logprob
    
    def beam(
            self,
            rnn_state,
            beam_size,
            max_len,
            att_features=None,
            att_tokens=None,
    ):
        assert rnn_state[0].shape[1] == 1
        device = rnn_state[0].device

        # init attention
        if att_features is None:
            att_features = ()
            att_tokens = ()
            att_masks = ()
            att_token_proj = ()
        else:
            assert isinstance(att_features, list) \
                or isinstance(att_features, tuple)
            att_masks = tuple(
                (toks == self.vocab.pad()).float() for toks in att_tokens
            )
            att_token_proj = tuple(
                self._make_projection(toks) for toks in att_tokens
            )

        # initialize beam
        beam = [BeamState(
            rnn_state[0].new_zeros(self.n_hidden),
            [s.squeeze(1) for s in rnn_state],
            [],
            [self.vocab.sos()],
            0.,
            None,
            False
        )]

        for t in range(max_len):
            if all(s.done for s in beam):
                break
            rnn_state = [
                torch.stack([s.rnn_state[i] for s in beam], dim=1)
                for i in range(len(beam[0].rnn_state))
            ]
            tokens = torch.LongTensor([
                [s.tokens[tt] if tt < len(s.tokens) else s.tokens[-1] for s in beam] 
                for tt in range(t+1)
            ]).to(device)
            decoder_state = DecoderState(
                torch.stack([s.feed for s in beam]),
                rnn_state,
                [torch.stack(
                    [s.hiddens[tt] if tt < len(s.hiddens) else s.hiddens[-1] for s in beam],
                dim=1) for tt in range(t)],
                tokens,
            )
            self_att_proj = self._make_projection(tokens)
            pred, feed, rnn_state, hidden, *_ = self.step(
                decoder_state,
                tuple(f.expand(f.shape[0], len(beam), f.shape[2]) for f in att_features),
                tuple(t.expand(t.shape[0], len(beam)) for t in att_tokens),
                tuple(m.expand(m.shape[0], len(beam)) for m in att_masks),
                tuple(p.expand(p.shape[0], len(beam), p.shape[2]) for p in att_token_proj),
                self_att_proj
            )

            logprobs = F.log_softmax(pred, dim=1)
            next_beam = []
            for i, row in enumerate(logprobs):
                row[self.vocab.copy()] = -np.inf
                scores, toks = row.topk(beam_size)
                if beam[i].done:
                    next_beam.append(beam[i])
                else:
                    for s, t in zip(scores, toks):
                        next_beam.append(BeamState(
                            feed[i, :],
                            [s[:, i, :] for s in rnn_state],
                            beam[i].hiddens + [hidden[:, i, :]],
                            beam[i].tokens + [t.item()],
                            beam[i].score + s,
                            beam[i],
                            t == self.vocab.eos()
                        ))
            next_beam = sorted(next_beam, key=lambda x: -x.score)
            beam = next_beam[:beam_size]

        return [s.tokens for s in beam]
