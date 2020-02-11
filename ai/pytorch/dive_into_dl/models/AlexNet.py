import time
import torch
from torch import nn, optim
import torchvision
import utils

device = torch.device('cuda' if torch.cuda.is_available() 
        else 'cpu')

# n = (n - k + 2 * p) / s + 1

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # (b, 1, 224, 224)
            nn.Conv2d(1, 96, 11, 4),
            # -> (b, 96, 54, 54),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # -> (b, 96, 26, 26)
            nn.Conv2d(96, 256, 5, 1, 2),
            # -> (b, 256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # -> (b, 256, 12, 12)
            nn.Conv2d(256, 384, 3, 1, 1),
            # -> (b, 384, 12, 12)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            # -> (b, 384, 12, 12) 
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            # -> (b, 256, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # -> (b, 256, 5, 5)
        )
        self.fc = nn.Sequential(
            # (b, 256*5*5)
            nn.Linear(256*5*5, 4096),
            # (b, 4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            # (b, 4096)
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
            # (b, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def train():
    net = AlexNet()
    print(net)
    
    batch_size = 128
    img_size = 224
    train_iter, test_iter = utils.load_data_fashion_mnist(
            batch_size, resize=img_size) 
    
    lr, num_epochs = 0.001, 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
            device, num_epochs)


if __name__ == '__main__':
    train()
