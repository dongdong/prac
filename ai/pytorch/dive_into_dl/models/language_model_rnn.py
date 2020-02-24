import torch
import utils
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus_indices, char_to_idx, idx_to_char, vocab_size = utils.load_data_jay_lyrics()
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), 
                          device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, 
                                         requires_grad=True))
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device,
                                         requires_grad=True))

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, )


def train():
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    #random_iter = True
    random_iter = False
    print('will use', device)
    utils.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, 
            vocab_size, device, corpus_indices, idx_to_char, char_to_idx, random_iter, 
            num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, 
            pred_len, prefixes)


def train_pytorch():
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
    model = utils.RNNModel(rnn_layer, vocab_size).to(device) 
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    print('will use', device)
    utils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, 
            corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr, 
            clipping_theta, batch_size, pred_period, pred_len, prefixes)


if __name__ == '__main__':
    #train()
    train_pytorch()

