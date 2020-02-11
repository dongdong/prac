import time
import torch
from torch import nn, optim
import utils

'''
VGG块的组成规律：
    连续使用数个相同的填充为1，形状为3X3的卷积层后，
    接上一个步幅为2，窗口形状为2X2的最大池化层
    卷积层保持输入的高和宽不变，而池化层则对其减半
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, 
                    kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels,
                    kernel_size=3, padding=1))
            blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units=4096, output=10):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        block = vgg_block(num_convs, in_channels, out_channels)
        net.add_module('vgg_block_' + str(i+1), block)

    fc  = nn.Sequential(
            utils.FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, output)
    )
    net.add_module("fc", fc)
    
    return net

def get_net():
    conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512),
            (2, 512, 512))
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096
    net = vgg(conv_arch, fc_features, fc_hidden_units)
    return net


def get_small_net():
    ratio = 8
    conv_arch = ((1, 1, 64//ratio), (1, 64//ratio, 128//ratio), 
            (2, 128//ratio, 256//ratio), (2, 256//ratio, 512//ratio),
            (2, 512//ratio, 512//ratio))
    fc_features = 512 * 7 * 7 // ratio
    fc_hidden_units = 4096 // ratio
    net = vgg(conv_arch, fc_features, fc_hidden_units)
    return net


def print_net(net):
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, blk in net.named_children():
        X= blk(X)
        print(name, 'output shape: ', X.shape)


def test():
    net = get_net()
    print_net(net)
    net = get_small_net()
    print_net(net)


def train():
    net = get_small_net()
    batch_size = 64
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, 
            resize=224)
    lr, num_epochs = 0.001, 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, 
            device, num_epochs)


if __name__ == '__main__':
    #test()
    train()


