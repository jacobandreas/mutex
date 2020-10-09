import torch
from torch import nn, optim
import torch.utils.data as torch_data
import torch.nn.functional as F
import itertools as it
import numpy as np
import random
from itertools import combinations, product
from mutex import EncDec, Vocab, batch_seqs, Mutex
from data import encode,  generate_fig2_exp, Oracle, collate, eval_format
from absl import app, flags
import sys
import os

FLAGS = flags.FLAGS
flags.DEFINE_integer("dim", 200, "trasnformer dimension")
flags.DEFINE_integer("n_layers", 1, "number of rnn layers")
flags.DEFINE_integer("n_batch", 1, "batch size")
flags.DEFINE_integer("n_epochs",50, "number of training epochs")
flags.DEFINE_integer("Nsample",100, "number of samples from py")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("temp", 1.0, "temperature for samplings")
flags.DEFINE_float("dropout", 0.05, "dropout")
flags.DEFINE_float("lamda", 0.1, "lambda")
flags.DEFINE_float("kl_lamda", 1.0, "extra lambda for kl")
flags.DEFINE_float("ent", 0.0, "qx|y entropy")
flags.DEFINE_string("save_model", "model.m", "model save location")
flags.DEFINE_integer("seed", 10, "random seed")
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("regularize", False, "apply regularization")
flags.DEFINE_bool("full_data", False, "full figure 2 experiments, otherwise color matching")

import hlog

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.dpi'] = 250

#input_symbols_list   = set(['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer'])


DEVICE = torch.device("cuda:0")

def pretrain(model, train_dataset, val_dataset):
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=FLAGS.n_batch, shuffle=True, collate_fn=collate
    )

    best_loss  = np.inf

    for i_epoch in range(2*FLAGS.n_epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        for inp, _ in train_loader:
            x = inp[:-1,:]
            pred, *extras = model(None, x.shape[0], x.to(DEVICE))
            output = pred.view(-1, len(model.vocab))
            loss = model.nllreduce(output,inp[1:, :].view(-1).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss    += loss.item() * inp.shape[1]
            train_batches += inp.shape[1]

        if (i_epoch + 1) % 2 != 0:
            continue

        curr_loss = train_loss / train_batches
        best_loss = min(best_loss, curr_loss)

        hlog.value("loss", curr_loss)
        hlog.value("best loss", best_loss)


    hlog.value("best loss", best_loss)



def train(model, train_dataset, val_dataset):
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)

    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=FLAGS.n_batch, shuffle=False,
        collate_fn=collate
    )

    best_f1  = -np.inf
    best_acc = -np.inf

    for i_epoch in range(FLAGS.n_epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        for inp, out in train_loader:
            nll = model(inp.to(DEVICE), out.to(DEVICE))
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()
            train_batches += 1

        if (i_epoch + 1) % 3 != 0 and i_epoch != FLAGS.n_epochs:
            continue

        with hlog.task(i_epoch):
            hlog.value("curr loss", train_loss / train_batches)
            acc, f1 = validate(model, val_dataset)
            hlog.value("acc", acc)
            hlog.value("f1", f1)
            best_f1 = max(best_f1, f1)
            best_acc = max(best_acc, acc)
            hlog.value("best_acc", best_acc)
            hlog.value("best_f1", best_f1)
            print()
    torch.save(model, f"seed_{FLAGS.seed}_"+ FLAGS.save_model)

    hlog.value("final_acc", acc)
    hlog.value("final_f1", f1)
    hlog.value("best_acc", best_acc)
    hlog.value("best_f1", best_f1)
    return acc, f1


def validate(model, val_dataset, vis=False):
    model.eval()
    hlog.value("qxy samples", model.sample_qxy(model.py.sample(20,model.MAXLEN_Y),temp=model.temp))
    first = True
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=FLAGS.n_batch, shuffle=True,
        collate_fn=collate
    )
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for inp, out in val_loader:
            pred, _ = model.sample(inp.to(DEVICE), temp=1.0, max_len=model.MAXLEN_Y, greedy=True)
            for i, seq in enumerate(pred):
                ref = out[:, i].detach().cpu().numpy().tolist()
                ref = eval_format(model.vocab_y, ref)
                pred_here = eval_format(model.vocab_y, pred[i])
                correct_here = pred_here == ref
                correct += correct_here
                tp_here = len([p for p in pred_here if p in ref])
                tp += tp_here
                fp_here = len([p for p in pred_here if p not in ref])
                fp += fp_here
                fn_here = len([p for p in ref if p not in pred_here])
                fn += fn_here
                total += 1
                if vis:
                    with hlog.task(total):
                        hlog.value("label", correct_here)
                        hlog.value("tp",tp_here)
                        hlog.value("fp",fp_here)
                        hlog.value("fn",fn_here)
                        inp_lst = inp[:, i].detach().cpu().numpy().tolist()
                        hlog.value("input", eval_format(model.vocab_x, inp_lst))
                        hlog.value("gold", ref)
                        hlog.value("pred", pred_here)


    acc = correct / total
    if tp+fp > 0:
        prec = tp / (tp + fp)
    else:
        prec=0
    rec = tp / (tp + fn)
    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    hlog.value("acc", acc)
    hlog.value("f1", f1)
    return acc, f1

def swap_io(items):
    return [(y,x) for (x,y) in items]

def main(argv):
    hlog.flags()

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)


    #input_symbols_list   = set(['red', 'yellow', 'green', 'blue', 'purple', 'pink', 'around', 'thrice', 'after'])
    input_symbols_list   = set(['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer'])
    output_symbols_list  = set(['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK'])

    study, test = generate_fig2_exp(input_symbols_list, output_symbols_list)

    vocab_x = Vocab()
    vocab_y = Vocab()

    if FLAGS.full_data:
        for sym in input_symbols_list:
            vocab_x.add(sym)
        for sym in output_symbols_list:
            vocab_y.add(sym)
        max_len_x = 7
        max_len_y = 9
    else:
        test, study  = study[3:4], study[0:3]
        for (x,y) in test+study:
            for sym in x:
                vocab_x.add(sym)
            for sym in y:
                vocab_y.add(sym)
        max_len_x = 2
        max_len_y = 2

    hlog.value("vocab_x\n", vocab_x)
    hlog.value("vocab_y\n", vocab_y)
    hlog.value("study\n", study)
    hlog.value("test\n", test)


    train_items, test_items = encode(study,vocab_x, vocab_y), encode(test,vocab_x, vocab_y)

#   outlist = list(output_symbols_list)

    oracle_py  = Oracle(train_items, test_items, DEVICE, dist="py",  vocab_x=vocab_x, vocab_y=vocab_y)
    oracle_px  = Oracle(train_items, test_items, DEVICE, dist="px",  vocab_x=vocab_x, vocab_y=vocab_y)
    oracle_qxy = Oracle(train_items, test_items, DEVICE, dist="qxy", vocab_x=vocab_x, vocab_y=vocab_y)

    model = Mutex(vocab_x,
                  vocab_y,
                  FLAGS.dim,
                  FLAGS.dim,
                  oracle_py,
                  max_len_x=max_len_x,
                  max_len_y=max_len_y,
                  copy=False,
                  n_layers=FLAGS.n_layers,
                  self_att=False,
                  dropout=FLAGS.dropout,
                  lamda=FLAGS.lamda,
                  kl_lamda=FLAGS.kl_lamda,
                  Nsample=FLAGS.Nsample,
                  temp=FLAGS.temp,
                  regularize=FLAGS.regularize,
                  ent=FLAGS.ent,
                 ).to(DEVICE)

    if FLAGS.regularize and not isinstance(model.px,Oracle):
        with hlog.task("pretrain px"):
            pretrain(model.px, train_items + test_items, test_items)
            for p in model.px.parameters():
                p.requires_grad = False


    with hlog.task("Initial Samples"):
        hlog.value("px samples\n",  "\n".join(model.sample_px(20)))
        hlog.value("py samples\n",  "\n".join(model.sample_py(20)))
        hlog.value("qxy debug samples\n", "\n".join(model.sample_qxy_debug(N=20)))
        hlog.value("qxy debug data\n", "\n".join(model.sample_qxy_debug_data(train_items + test_items)))
#         hlog.value("qxy samples", "\n".join(model.sample_qxy(model.py.sample(20,max_len),temp=model.temp)))
#         hlog.value("qxy samples (gumbel)", "\n".join(model.sample_qxy_gumbel(model.py.sample(20,max_len),temp=model.temp)))

#     if not isinstance(model.qxy,Oracle):
#         train(model.qxy, swap_io(train_items) + swap_io(test_items), swap_io(test_items))
#     if not isinstance(model.pyx,Oracle):
#         train(model.pyx, train_items + test_items, test_items)
#         for param in model.pyx.parameters():
#             param.requires_grad = False

    with hlog.task("train model"):
        acc, f1 = train(model, train_items, test_items)

    with hlog.task("Final Samples"):
        hlog.value("px samples\n", "\n".join(model.sample_px(20)))
        hlog.value("py samples\n", "\n".join(model.sample_py(20)))
        hlog.value("qxy debug samples\n", "\n".join(model.sample_qxy_debug(N=20)))
        hlog.value("qxy debug data\n", "\n".join(model.sample_qxy_debug_data(train_items + test_items)))
        hlog.value("qxy samples (gumbel)\n", "\n".join(model.sample_qxy_gumbel(model.py.sample(20,max_len_y),temp=model.temp)))
        #hlog.value("qxy samples", "\n".join(model.sample_qxy(model.py.sample(20,max_len),temp=model.temp)))

    if FLAGS.regularize:
        losses = pd.DataFrame(model.loss_container)
        figure = sns.lineplot(data=losses, dashes=False).figure
        figure.savefig(f"{FLAGS.seed}_plot.png")

    with hlog.task("train evaluation"):
        validate(model, train_items, vis=True)

    with hlog.task("test evaluation"):
        validate(model, test_items, vis=True)

if __name__ == "__main__":
    app.run(main)
