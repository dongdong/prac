import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time
import zipfile
import random
import math

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, X):
        return X.view(X.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def sgd(params, lr, batch_size):
    #print("sgd, lr %f, batch_size, %d" % 
    #      (lr, batch_size))
    for param in params:
        param.data -= lr * param.grad / batch_size
    
    
def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt',
        'trouser',
        'pullover',
        'dress',
        'coat',
        'sandal',
        'shirt',
        'sneaker',
        'bag',
        'ankle boot',
    ]
    return [text_labels[int(i)] for i in labels]

    
def load_data_fashion_mnist(batch_size=256, resize=None):
    mnist_data_path = '~/Datasets/FashionMNIST'
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=mnist_data_path,
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=mnist_data_path,
        train=False,
        download=True,
        transform=transform
    )

    num_workers = 2
    train_iter = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += ((net(X.to(device)).argmax(
                    dim=1) == y.to(device)).float()
                            .sum().cpu().item())
                net.train()
            else:
                if ('is_training' in 
                    net.__code__.co_varnames):
                    acc_sum += ((net(X, is_training=False)
                                .argmax(dim=1) == y)
                               .float().sum().item())
                else:
                    acc_sum += ((net(X).argmax(dim=1) 
                                 == y).float().sum()
                                .item())
            n += y.shape[0]
    return acc_sum / n 


def train_ch5(net, train_iter, test_iter, batch_size, 
    optimizer, device, num_epochs):
    net = net.to(device)
    print("train on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum = 0.0, 0.0
        n, start = 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += ((y_hat.argmax(dim=1) == y)
                             .sum().cpu().item())
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        print(('epoch %d, loss %.4f, train acc %.3f, ' + 
             'test acc %.3f, time %.1f') % (
                 epoch + 1, train_l_sum / batch_count, 
                 train_acc_sum / n, test_acc, 
                 time.time() - start))

        
def load_data_jay_lyrics():
    data_path = '../data/jaychou_lyrics.txt.zip'
    with zipfile.ZipFile(data_path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    #corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) 
                        for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] 
                  for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, 
                     num_steps, device=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    if device is None:
        device = torch.device('cuda' if 
                torch.cuda.is_available() else 'cpu')
        
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in 
                 batch_indices]
        yield torch.tensor(X, dtype=torch.float32, 
                           device=device),  torch.tensor(
                Y, dtype=torch.float32, device=device)
        

def data_iter_consecutive(corpus_indices, batch_size,
                         num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if 
                torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, 
                                 dtype=torch.float32,
                                 device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    #print(data_len, batch_len)
    indices = corpus_indices[0: batch_size * batch_len
                            ].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

        
def one_hot(x, n_class, dtype=torch.float32):
    # x shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype,
                     device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len)
    # output: seq_len elsements of (batch, n_class)
    return [one_hot(X[:, i], n_class) 
            for i in range(X.shape[1])]


def predict_rnn(prefix, num_chars, rnn, params, 
                init_rnn_state, num_hiddens, vocab_size,
               device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], 
                device=device), vocab_size)
        # 计算输出，更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
            
         
def train_and_predict_rnn(rnn, get_params, init_rnn_state,
                         num_hiddens, vocab_size, 
                         device, corpus_indices, 
                         idx_to_char, char_to_idx,
                         is_random_iter, num_epochs,
                         num_steps, lr, clipping_theta, 
                         batch_size, pred_period, pred_len,
                         prefixes):
    print('train_and_predict_rnn. batch_size: %d, num_hidden: %d, lr: %f' % 
        (batch_size, num_hiddens, lr))
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, 
                                   num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, 
                                 batch_size, num_steps,
                                 device)
        for X, Y in data_iter:
            if is_random_iter:
                # 如果随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, 
                                       num_hiddens, 
                                       device)
            else:
                # 否则需要使用detach函数从计算图分离隐藏状态
                # 这样为了使模型参数的梯度计算只依赖一次迭代
                # 防止梯度计算开销太大
                for s in state:
                    s.detach_()
            
            inputs = to_onehot(X, vocab_size)
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(outputs, y.long())
            
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
            l.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, l)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' 
                  % (epoch + 1, math.exp(l_sum / n), 
                 time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len,
                    rnn, params, init_rnn_state, 
                    num_hiddens, vocab_size, device,
                    idx_to_char, char_to_idx))
                
         
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (
                2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None
    
    def forward(self, inputs, state): 
        # inputs: (batch, seq_len)
        X = to_onehot(inputs, self.vocab_size)
        # X: (seq_len, batch, vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        # Y: (seq_len, batch, hidden)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state
    
 
def predict_rnn_pytorch(prefix, num_chars, model, 
                        vocab_size, device, idx_to_char,
                       char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device
                        ).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device),
                        state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
            
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens,
        vocab_size, device, corpus_indices, idx_to_char,
        char_to_idx, num_epochs, num_steps, lr, 
        clipping_theta, batch_size, pred_period, pred_len,
        prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, 
                batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), 
                            state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)
            # output shape: 
            # (num_steps * batch_size, vocab_size)
            y = torch.transpose(Y, 0, 1).contiguous(
                    ).view(-1)
            # shape: batch * num_steps 与y一致
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), 
                         clipping_theta,
                         device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec'
                 % (epoch + 1, perplexity, 
                  time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn_pytorch(prefix,
                        pred_len, model, vocab_size,
                        device, idx_to_char, char_to_idx)
                     )
                
