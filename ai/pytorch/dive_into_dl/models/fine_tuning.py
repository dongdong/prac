import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_iter(batch_size=128):
    data_dir = 'cache/hotdog'
    # ！！！在使用预训练模型时，一定要和预训练时作同样的预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    train_augs = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    test_augs = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize,
    ])
    train_iter = DataLoader(
            ImageFolder(os.path.join(data_dir, 'hotdog/train'), 
                transform=train_augs), batch_size, shuffle=True)
    test_iter = DataLoader(
            ImageFolder(os.path.join(data_dir, 'hotdog/test'), 
                transform=test_augs), batch_size)
    return train_iter, test_iter


def test_data():
    train_iter, test_iter = get_data_iter()
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break


def train_fine_tune(train_iter, test_iter, num_epochs=5):
    pretrained_net = models.resnet18(pretrained=True)
    # previous: (512, 1000)
    pretrained_net.fc = nn.Linear(512, 2)
    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, 
            pretrained_net.parameters())
    lr = 0.01
    optimizer = optim.SGD([
            {'params': feature_params},
            {'params': pretrained_net.fc.parameters(), 'lr': lr*10}
    ], lr=lr, weight_decay=0.001)

    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, pretrained_net, loss, optimizer,
            device, num_epochs)


def train_from_scratch(train_iter, test_iter, num_epochs=5):
    scratch_net = models.resnet18(pretrained=False, num_classes=2)
    lr = 0.1
    optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, scratch_net, loss, optimizer,
            device, num_epochs)
    

def do_train():
    train_iter, test_iter = get_data_iter()
    #train_fine_tune(train_iter, test_iter)
    train_from_scratch(train_iter, test_iter)


if __name__ == '__main__':
    #test_data()
    do_train()
