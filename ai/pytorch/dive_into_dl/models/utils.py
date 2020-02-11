import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time

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
