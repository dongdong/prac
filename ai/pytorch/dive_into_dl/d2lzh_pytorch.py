import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
import zipfile


def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(),
            y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    
def semilogy(x_vals, y_vals, x_labels, y_labels, 
            x2_vals=None, y2_vals=None, legend=None,
            figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
        
mnist_data_path = '~/Datasets/FashionMNIST'
mnist_train = torchvision.datasets.FashionMNIST(
    root=mnist_data_path,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
mnist_test = torchvision.datasets.FashionMNIST(
    root=mnist_data_path,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

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

def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), 
                           figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
def load_data_fashion_mnist(batch_size=256):
    '''
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    '''
    num_workers = 0
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

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
    
def sgd(params, lr, batch_size):
    #print("sgd, lr %f, batch_size, %d" % 
    #      (lr, batch_size))
    for param in params:
        param.data -= lr * param.grad / batch_size

        
def load_data_jay_lyrics():
    data_path = 'data/jaychou_lyrics.txt.zip'
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
