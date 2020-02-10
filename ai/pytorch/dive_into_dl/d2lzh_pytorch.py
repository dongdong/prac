import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms


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
