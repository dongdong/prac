import time
import torch
from torch import nn
import utils 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_net():
    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        utils.FlattenLayer(),
        nn.Linear(16*4*4, 120),
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    return net


def train():
    net = get_net()
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(
        batch_size=batch_size)
    lr, num_epochs = 0.001, 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
        device, num_epochs)


if __name__ == "__main__":
    train()


