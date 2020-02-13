''' DensNet - 稠密连接网络
与ResNet的主要区别在于，DenseNet的输出不是相加，而是在通道维上连结
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    ''' 批量归一化，激活，卷积
    '''
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                padding=1)
    ) 
    return blk


class DenseBlock(nn.Module):
    ''' 稠密块由多个conv_block组成
        每块使用相同的输出通道数，
        每块的输入和输出在通道维上连结
    '''
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels
        
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(in_channels, out_channels):
    ''' 过渡层，控制模型复杂度
        通过1x1卷积层来减小通道数，
        并使用步幅为2的平均池化层减半高和宽
    '''
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )    
    return blk


def get_net():
    ''' DenseNet
    '''
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 4个稠密块，每个稠密块4个卷积层
    # 稠密块卷积层通道数32， 每个稠密块将增加128个通道
    # 使用过渡层来减半高和宽，并减半通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module('DenseBlock_%d' % i, DB)
        # 上一个稠密块的输出通道
        num_channels = DB.out_channels
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, 
                    transition_block(num_channels, 
                            num_channels // 2))
            num_channels = num_channels // 2
    
    # 全局池化层和全连接层
    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", utils.GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(
            utils.FlattenLayer(),
            nn.Linear(num_channels, 10)
    ))
    return net


def print_DenseBlock():
    X = torch.rand(4, 3, 8, 8)
    print(X.shape)
    blk = DenseBlock(2, 3, 10)
    print(blk)
    Y = blk(X)
    print(Y.shape)
    blk = transition_block(23, 10)
    print(blk)
    Y = blk(Y)
    print(Y.shape)
    

def print_net():
    net = get_net()
    print(net)
    X = torch.rand((1, 1, 96, 96))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, 'output shape\t', X.shape)


def train():
    net = get_net()
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(
            batch_size, resize=96)
    lr, num_epochs = 0.001, 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size,
            optimizer, device, num_epochs)


if __name__ == '__main__':
    #print_DenseBlock()
    #print_net()
    train()




