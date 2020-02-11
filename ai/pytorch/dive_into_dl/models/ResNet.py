''' ResNet - 残差网络
- 残差块通过跨层的数据通道从而能够训练出有效的神经网络
'''

import torch
from torch import nn, optim
import torch.nn.functional as F
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Residual(nn.Module):
    '''' 残差块
        - 首先，有2个有相同输出通道的3x3卷积层，
            - 每个卷积层后接一个批量归一化层和ReLU激活函数
        - 然后，将输入跳过这两个卷积层直接加在最后的ReLU激活函数前
            - 要求输入输出形状一样才可以相加
            - 通过1x1卷积层将输入变换成需要的形状
    '''
    def __init__(self, in_channels, out_channels, 
            use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):   
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, 
        first_block=False):
    if first_block:
        # 第一个模块通道数与输出通道数一致
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):    
        if i == 0 and not first_block:
            # 宽高减半
            blk.append(Residual(in_channels, out_channels, 
                    use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    
    return nn.Sequential(*blk)


def get_net():
    ''' ResNet
    '''
    # (b, 1, 224, 224)
    # (224 - 7 + 3 * 2) / 2 + 1 = 112
    # (112 - 3 + 1 * 2) / 2 + 1 = 56
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # (b, 64, 56, 56)
    net.add_module('resnet_block1', resnet_block(64, 64, 2, 
            first_block=True))
    # (b, 64, 56, 56)
    net.add_module('resnet_block2', resnet_block(64, 128, 2)) 
    # (b, 128, 28, 28)
    net.add_module('resnet_block3', resnet_block(128, 256, 2)) 
    # (b, 256, 14, 14)
    net.add_module('resnet_block4', resnet_block(256, 512, 2)) 
    # (b, 512, 7, 7)
    
    net.add_module('global_avg_pool', utils.GlobalAvgPool2d())
    # (b, 512, 1, 1)
    net.add_module('fc', nn.Sequential(utils.FlattenLayer(),
            nn.Linear(512, 10)))
    # (b, 10)

    return net


def print_net():
    net = get_net()
    print(net)
    X = torch.rand(1, 1, 224, 224)
    for name, layer in net.named_children():
        X = layer(X)
        print(name, 'output shape:\t', X.shape)


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
    #print_net()
    train()
