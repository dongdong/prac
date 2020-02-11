import time
import torch
from torch import nn
import utils 


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            # (b, 1, 28, 28)
            nn.Conv2d(1, 6, 5), # in, out, kernel
            # -> (b, 6, 24, 24)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            # -> (b, 6, 12, 12)
            nn.Conv2d(6, 16, 5),
            # -> (b, 16, 8, 8)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
            # -> (b, 16, 4, 4)
        )
        self.fc = nn.Sequential(
            # -> (b, 16*4*4)
            nn.Linear(16*4*4, 120),
            # -> (b, 120)
            nn.Sigmoid(),
            nn.Linear(120, 84),
            # -> (b, 84)
            nn.Sigmoid(),
            nn.Linear(84, 10)
            # -> (b, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def train():
    device = torch.device('cuda' if torch.cuda.is_available() 
        else 'cpu')
    net = LeNet()
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(
        batch_size=batch_size)
    lr, num_epochs = 0.001, 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
        device, num_epochs)


if __name__ == "__main__":
    train()


