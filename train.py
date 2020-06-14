#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.utils.data as torch_data
import itertools as it

def tensorize(data):
    return [
        (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        for x, y in data
    ]

data_perm = tensorize([
    ([1, 0, 0], [0, 1, 0]),
    ([0, 1, 0], [1, 0, 0]),
    ([0, 0, 1], [0, 0, 1])
])
train_perm = data_perm[:2]
test_perm = data_perm[2:]
posterior_perm = [y for x, y in data_perm]

data_comp = tensorize([
    ([1, 0, 1, 0], [0, 1, 0, 1]),
    ([1, 0, 0, 1], [0, 1, 1, 0]),
    ([0, 1, 1, 0], [1, 0, 0, 1]),
    ([0, 1, 0, 1], [1, 0, 1, 0]),
])
train_comp = data_comp[:2]
test_comp = data_comp[2:]
posterior_comp = [y for x, y in data_comp]

data_len = tensorize([
    (
        [1. for _ in range(i)] + [0. for _ in range(10-i)],
        [1. for _ in range(i)] + [0. for _ in range(10-i)]
    )
    for i in range(10)
])
train_len = data_len[:5]
test_len = data_len[5:]
posterior_len = [y for x, y in data_len]

class Enc(nn.Module):
    def __init__(self, d_y, d_hid, d_x):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_y, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, d_x),
            #nn.Linear(d_y, d_x),
            #nn.Softmax()
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.layers(y)

class Dec(nn.Module):
    def __init__(self, d_x, d_hid, d_y):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_x, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, d_y),
            #nn.Linear(d_x, d_y),
            #nn.Softmax()
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

def main():
    train_loader = torch_data.DataLoader(train_comp, shuffle=True)
    posterior_loader = torch_data.DataLoader(posterior_comp, shuffle=True)
    test_loader = torch_data.DataLoader(test_comp, shuffle=False)
    dim = 3
    loss_fn = nn.BCELoss()

    for use_y in (True, False,):
        mean_test_loss = 0
        for restart in range(10):
            enc = Enc(dim, 64, dim)
            dec = Dec(dim, 64, dim)
            prior = nn.Parameter(torch.zeros(1, dim))
            opt = optim.Adam(
                it.chain(
                    enc.parameters(), dec.parameters(), [prior]), lr=0.01
            )
            for i in range(500):
                for (x_, y_), y in zip(train_loader, posterior_loader):
                    nlp_yx = loss_fn(dec(x_), y_)
                    loss = nlp_yx 
                    if use_y:
                        x = enc(y)
                        #x = x + torch.normal(torch.zeros_like(x), 0.1)
                        nlp_x = loss_fn(torch.exp(prior), x.detach()) \
                                + loss_fn(x, torch.exp(prior).detach())
                        nlp_y = loss_fn(dec(x), y)
                        loss += 0.1 * (nlp_x + nlp_y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            test_loss = 0
            #print()
            #print(prior)
            for x, y in train_loader:
                #print(x.detach().cpu().numpy(), y.detach().cpu().numpy(),
                #        enc(y).detach().cpu().numpy(),
                #        dec(x).detach().cpu().numpy())
                pass
            for x, y in test_loader:
                #print(x.detach().cpu().numpy(), y.detach().cpu().numpy(),
                #        enc(y).detach().cpu().numpy(),
                #        dec(x).detach().cpu().numpy())
                nlp_yx = loss_fn(dec(x), y)
                test_loss += nlp_yx.item()
            #print()
            mean_test_loss += test_loss
        print()
        print(mean_test_loss / 10)

if __name__ == "__main__":
    main()
