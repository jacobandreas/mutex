import torch
from torch import nn, optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import itertools as it
import numpy as np
import random
from itertools import combinations, product
import sys
from src import batch_seqs

def encode(data,vocab):
    encoded = []
    for (inp,out) in data:
        encoded.append(( [vocab.sos()]  + vocab.encode(inp) + [vocab.eos()], [vocab.sos()] + vocab.encode(out) + [vocab.eos()]))
    return encoded

def eval_format(vocab, seq):
    if vocab.eos() in seq:
        seq = seq[:seq.index(vocab.eos())+1]
    seq = seq[1:-1]
    return vocab.decode(seq)

def collate(batch):
    inp, out = zip(*batch)
    inp = batch_seqs(inp)
    out = batch_seqs(out)
    return inp, out

f1  = lambda w:  w + w + w
f2  = lambda w1, w2: w1 + w2 + w1
f3  = lambda w1, w2: w2 + w1
get = lambda vals, hsh: [hsh[val] for val in vals]

def unary(f, w, colormap, fmap):
    return (w + [fmap[f]], get(f(w),colormap))

def binary(f, w1, w2, colormap, fmap):
    return (w1 + [fmap[f]] + w2, get(f(w1,w2),colormap))

def binary_mapped(f, p1, p2, fmap):
    w1, o1 = p1
    w2, o2 = p2
    return (w1 + [fmap[f]] + w2, f(o1,o2))

def samplef(fs,words,colormap,fmap):
    f = random.choice(fs)
    if f == f1:
        w = random.choice(tuple(words))
        return unary(f, [w], colormap, fmap)
    else:
        w1, w2 = random.choice(list(product(words,words)))
        return binary(f, [w1], [w2], colormap, fmap)
    
def generate_fig2_exp(input_symbols, output_symbols):
        words     = set(random.sample(input_symbols,4))
        colors    = set(random.sample(output_symbols,4))
        colormap  = dict(zip(words, colors))
        fnames    = random.sample(input_symbols - words,3)
        fmap      = dict(zip([f1,f2,f3], fnames))
        print("color map: ", colormap)
        print("function names: ", fnames)
        
        trn,tst  = [],[]
        #Primitives
        for (i,w) in enumerate(words):
            trn.append(([w],get([w],colormap)))
            
            
        combs = set(combinations(words, r=2))
        #Function 1 : x f1 -> X X X 
        trnwords = set(random.sample(words,2))
        tstwords = set(random.sample(words-trnwords,2)) 
        for w in trnwords:
            trn.append(unary(f1,[w],colormap, fmap)) 
        for w in tstwords:
            tst.append(unary(f1,[w],colormap, fmap)) 
        
        #Function 2 : x f2 y-> X Y X 
        trnpairs = set(random.sample(combs,2))
        tstpairs = set(random.sample(combs-trnpairs,2)) 
        for (w1,w2) in trnpairs:
            trn.append(binary(f2,[w1],[w2],colormap, fmap))
        for (w1,w2) in tstpairs:
            tst.append(binary(f2,[w1],[w2],colormap, fmap))
                       
        #Function 3 : x f3 y-> Y X
        trnpairs = set(random.sample(combs,2))
        tstpairs = set(random.sample(combs-trnpairs,2)) 
        for (w1,w2) in trnpairs:
            trn.append(binary(f3, [w1], [w2], colormap, fmap))
        for (w1,w2) in tstpairs:
            tst.append(binary(f3, [w1], [w2], colormap, fmap))
            
        #Study Compositions
        for i in range(2):
            w1 = random.choice(tuple(words))
            p1 = ([w1], [colormap[w1]])
            center  = random.choice((f2,f3))
            p2 = samplef((f1,),words,colormap,fmap)
            trn.append(binary_mapped(center,p1,p2,fmap))
        
        # Order of The Operations: fother (*) always before fcenter (+)
        if random.random() > 0.5:
            fcenter,fother = f2,f3
        else:
            fcenter,fother = f3,f2

        for i in range(2): 
            w1 = random.choice(tuple(words))
            p1 = ([w1], [colormap[w1]])
            center = fcenter
            p2 = samplef((fother,),words,colormap,fmap)
            if i == 1: p2, p1 = p1,p2
            trn.append(binary_mapped(center,p1,p2,fmap))
        
        #Test Compositions
        for i in range(2):
            w1 = random.choice(tuple(words))
            p1 = ([w1], [colormap[w1]])
            center  = random.choice((f2,f3))
            p2 = samplef((f1,),words,colormap,fmap)
            tst.append(binary_mapped(center,p1,p2,fmap))
        
        w1 = random.choice(tuple(words))
        p1 = ([w1], [colormap[w1]])
        center = fcenter
        p2 = samplef((fother,),words,colormap,fmap)
        if random.random() > 0.5: p2, p1 = p1,p2
        tst.append(binary_mapped(center,p1,p2,fmap))
                
        for i in range(2):
            p1 = samplef((f1,),words,colormap,fmap)
            center  = fcenter
            p2 = samplef((fother,),words,colormap,fmap)
            if random.random() > 0.5: p2, p1 = p1,p2
            tst.append(binary_mapped(center,p1,p2,fmap))
            
        return trn,tst



class Oracle:
    def __init__(self, 
             vocab, 
             train_items,
             test_items,
             device,
             dist="px",
        ):
        self.vocab = vocab
        self.train_items = train_items
        self.test_items = test_items
        self.all_items  = train_items + test_items
        self.device = device
        if dist=="py":
            def oracle(batch, max_len, **kwargs):
                ys = []
                data = random.choices(self.all_items,k=batch)
                ys = [y for _,y in data]
                return batch_seqs(ys).to(self.device)
        elif dist=="px": 
            def oracle(batch, max_len, **kwargs):
                xs = []
                data = random.choices(self.all_items,k=batch)
                xs = np.unique(np.array([x for x,_ in data]), axis=0).tolist()
                batch = len(xs)
                logprobs = torch.log(torch.Tensor([1/len(self.all_items)]*batch).to(self.device))
                return batch_seqs(xs).to(self.device), logprobs     
                
        elif dist=="qxy":
            def oracle(ys, max_len, **kwargs):
                xlist, ylist = zip(*(self.all_items))
                xlist = list(xlist)
                ylist = list(ylist)
                xs = []
                for row in ys.detach().cpu().numpy().transpose():
                    row = list(row)
                    symbols = eval_format(vocab, row)
                    encoded = [vocab.sos()]  +  vocab.encode(symbols) + [vocab.eos()]
                    xs.append(xlist[ylist.index(encoded)])
                return batch_seqs(xs).to(self.device), None
            
        self.sample = oracle