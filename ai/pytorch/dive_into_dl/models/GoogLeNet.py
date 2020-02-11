'''
GoogLeNet - 吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inception(nn.Module):
    ''' GoogLeNet中的基础卷积块Inception块，有4条并行的线路
        前三条线路使用窗口1x1，3x3和5x5的卷积层来抽取不同空间尺度下的信息，
        其中，中间两个先对输入做1x1卷积来减少输入通道数
        第四条线路使用3x3最大池化层，后接1x1卷积层来改变通道数
        4条线路都使用了合适的填充来使输入与输出的高和宽一致
    '''
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


def get_net():
    ''' 主体卷积部分使用5个模块
        每个模块之间使用步幅为2的3x3池化层来减少输出高宽
    '''
    # (b, 1, 96, 96)
    # (96 - 7 + 3 * 2) / 2 + 1 = 47
    # (47 - 3 + 1 * 2) / 2 + 1 = 24
    b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # (b, 64, 24, 24)
    # (24 - 3 + 1 * 2) + 1 = 24
    # (24 - 3 + 1 * 2) / 2 + 1 = 12
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # (b, 192, 12, 12)
    # 64 + 128 + 32 + 32 = 256
    # 128 + 192 + 96 + 64 = 480
    # (12 -3 + 1 * 2) / 2 + 1  = 6
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # (b, 480, 6, 6)
    # 256 + 320 + 128 + 128 = 832
    # (6 - 3 + 1 * 2) / 2 + 1 = 3
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # (b, 832, 3, 3)
    # 384 + 384 + 128 + 128 = 1024
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        utils.GlobalAvgPool2d()
    )
    # (b, 1024, 1, 1)
    net = nn.Sequential(b1, b2, b3, b4, b5,
            utils.FlattenLayer(), 
            nn.Linear(1024, 10))

    return net


def print_net():
    net = get_net()
    print(net)
    X = torch.rand(1, 1, 96, 96)
    for blk in net.children():
        X = blk(X)
        print('output shape: ', X.shape)


def train():
    net = get_net()
    batch_size = 128
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, 
            resize=96)
    lr, num_epochs = 0.001, 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device,
            num_epochs)


if __name__ == "__main__":
    #print_net()
    train()




