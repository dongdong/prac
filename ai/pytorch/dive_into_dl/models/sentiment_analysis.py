import os
import collections
import random
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_imdb(folder='train'):
    data_root = './cache/imdb/aclImdb'
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    ''' data: list of [string, label]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    max_l = 500
    def pad(x):
        l = len(x)
        return x[:max_l] if l > max_l else x + [0] * (max_l - l)
    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) 
            for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def get_data_iter(train_data, test_data, vocab, batch_size=64):
    train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
    test_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)
    return train_iter, test_iter


class BiRNN(nn.Module):
    '''bidireactional LSTM
    '''
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens,
                num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)
    def forward(self, inputs):
        # inputs: (batch, seq_len)
        # embedings: (seq_len, batch, embed_size) 
        embeddings = self.embedding(inputs.permute(1, 0))
        # output, (h, c)
        # output: (seq_len, batch, 2 * hidden_num)
        outputs, _ = self.encoder(embeddings) 
        # encoding: (batch, 4*hidden_num)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        # outs: (batch, 2)
        outs = self.decoder(encoding)
        return outs


def get_pretrained_glove_embeddings():
    cache_dir = './cache/glove'
    glove = Vocab.GloVe(name='6B', dim=100, cache=cache_dir)
    return glove


def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])    
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 0
    if oov_count > 0:
        print("there are %d oov words.")
    return embed

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    pass


def predict_sentiment(net, vocab, sentence):
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'


def evaluate(net, vocab):
    sentences = [
        'this movie is so great',
        'this movie is not so bad',
        'I think this movie is terrific',
        'I believe this is a terrible movie',
        'I can not get this movie',
        'this movie is not my type',
    ]
    for sentence in sentences:
        ret = predict_sentiment(net, vocab, [word for word in sentence.split(' ')])
        print("%s \t==> %s" % (ret, sentence))


def get_data():
    train_data, test_data = read_imdb('train'), read_imdb('test')
    print("train size: %d, test size: %d" % (len(train_data), len(test_data)))
    vocab = get_vocab_imdb(train_data)
    print("# words in vocab: ", len(vocab))
    train_iter, test_iter = get_data_iter(train_data, test_data, vocab)
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
    print('# batches: ', len(train_iter))
    return train_iter, test_iter, vocab


def do_train_rnn():
    train_iter, test_iter, vocab = get_data()
    embed_size, num_hiddens, num_layers = 100, 100, 2
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    glove_vocab = get_pretrained_glove_embeddings()
    net.embedding.weight.data.copy_(
            load_pretrained_embedding(vocab.itos, glove_vocab))
    net.embedding.weight.requires_grad = False

    lr, num_epochs = 0.01, 10
    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr)
    loss = nn.CrossEntropyLoss()

    net = net.to(device)
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
    evaluate(net, vocab)

   
# TEXTCNN

def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    return torch.stack([corr1d(x, k) for x, k in zip(X, K)]).sum(dim=0)


# max-over-time pooling
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x): 
        # x: (batch, channel, seq_len)
        # ret: (batch, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2*embed_size,
                    out_channels=c, kernel_size=k))
    def forward(self, inputs):
        # (batch, seq_len, 2*embed_size)
        embeddings = torch.cat((self.embedding(inputs),
                                self.constant_embedding(inputs)), dim=2)
        # (batch, 2*embed_size, seq_len)
        embeddings = embeddings.permute(0, 2, 1) 
        # conv: (batch, channel, seq_len)
        # poll: (batch, channel, 1)
        # squeeze: (batch, channel) 
        # cat: (batch, n*channels)  
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1)
                for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def test_conv1d():
    X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
    print(corr1d(X, K))

    X = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5 ,6, 7, 8],
    ])
    K = torch.tensor([[1, 2], [3 ,4], [-1, -3]])
    # Y[0] = 1 * 0 + 2 * 1 + 3 * 1 + 4 * 2 + (-1) * 2 + (-3) * 3
    #      = 0 + 2 + 3 + 8 - 2 - 9 
    #      =  2
    print(corr1d_multi_in(X, K))


def do_train_textcnn():
    train_iter, test_iter, vocab = get_data()
    
    embed_size, kernel_size, num_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(vocab, embed_size, kernel_size, num_channels)
    
    glove_vocab = get_pretrained_glove_embeddings()
    net.embedding.weight.data.copy_(
            load_pretrained_embedding(vocab.itos, glove_vocab))
    net.constant_embedding.weight.data.copy_(
            load_pretrained_embedding(vocab.itos, glove_vocab))
    net.constant_embedding.weight.requires_grad = False
    
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=lr)
    loss = nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
    evaluate(net, vocab)


if __name__ == '__main__':
    #do_train_rnn()
    #test_conv1d()
    do_train_textcnn()






