import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import utils

max_window_size = 5
batch_size = 512
embed_size = 100
num_workers = 2 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_raw_data():
    train_data_path = '../data/ptb/ptb.train.txt'
    with open(train_data_path, 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
    return raw_dataset


def get_discard_fn(counter, idx_to_token, num_tokens):
    def discard(idx):
        return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / counter[idx_to_token[idx]] * num_tokens)
    return discard    


def process_raw_data(raw_dataset):
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] 
                for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])    
    print("# tokens: %d" % num_tokens)
    
    discard = get_discard_fn(counter, idx_to_token, num_tokens)
    subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
    print("# tokens after discard: %d" % sum([len(st) for st in subsampled_dataset]))
    
    return subsampled_dataset, idx_to_token, token_to_idx, counter

    
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), 
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def test_get_centers_and_contexts():
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('dataset', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('center', center, 'has contexts', context)


def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                i, neg_candidates = 0, random.choices(population, 
                        sampling_weights, k=int(1e5)) 
            neg, i = neg_candidates[i], i + 1
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives   


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])
    def __len__(self):
        return len(self.centers)


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1),
            torch.tensor(contexts_negatives),
            torch.tensor(masks),
            torch.tensor(labels))   


def get_data():
    raw_data = read_raw_data()
    proc_data, idx_to_token, token_to_idx, counter = process_raw_data(raw_data)
    all_centers, all_contexts = get_centers_and_contexts(proc_data, max_window_size)
    sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
    all_negatives = get_negatives(all_contexts, sampling_weights, 5)
    
    dataset = MyDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True, 
            collate_fn=batchify, num_workers=num_workers) 
    return data_iter, idx_to_token, token_to_idx
    

def test_get_data():
    data_iter, _, _ = get_data()
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], 
                batch):
            print(name, 'shape:', data.shape)
        break


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, 
                reduction='none', weight=mask)
        return res.mean(dim=1)


def test_loss_fn():
    def sigmd(x):
        return -math.log(1 / (1 + math.exp(-x)))
    pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    loss = SigmoidBinaryCrossEntropyLoss()
    print(loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)) 
    print("%.4f, %.4f" % (
            (sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4,
            (sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))


def get_similar_tokens(query_token, k, embed, idx_to_token, token_to_idx):
    W = embed.weight.data
    x= W[token_to_idx[query_token]]
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * 
            torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k+1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:
        print('\tcosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))


def similarity_evaluate(net, idx_to_token, token_to_idx):
    words = ['chip', 'china', 'england', 'woman', 'king']
    for word in words:
        print('word: ', word)
        get_similar_tokens(word, 3, net[0], idx_to_token, token_to_idx)
        get_similar_tokens(word, 3, net[1], idx_to_token, token_to_idx)


def train(net, data_iter, loss, lr, num_epochs, idx_to_token, token_to_idx):
    print('train on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            pred = skip_gram(center, context_negative, net[0], net[1]) 
            l = (loss(pred.view(label.shape), label, mask) * mask.shape[1] / 
                    mask.float().sum(dim=1)).mean()           
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, l_sum / n, 
                time.time() - start))
        if (epoch + 1) % 5 == 0:
            similarity_evaluate(net, idx_to_token, token_to_idx)


def do_train():
    lr = 0.01
    num_epochs = 100 
    data_iter, idx_to_token, token_to_idx = get_data()
    net = nn.Sequential(
            nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
            nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),
    )
    loss = SigmoidBinaryCrossEntropyLoss()

    train(net, data_iter, loss, lr, num_epochs, idx_to_token, token_to_idx)


def test():
    get_data()
    test_get_centers_and_contexts()
    test_get_data()
    test_loss_fn()


def get_pretrained_glove_embeddings():
    import torchtext.vocab as vocab
    cache_dir = './cache/glove'
    glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
    return glove


def knn(W, x, k):
    cos = torch.matmul(W, x.view((-1, ))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]


def get_similar_tokens_knn(query_token, k, embed):
    topk, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))


def test_pretrained_word2vec():
    glove = get_pretrained_glove_embeddings()
    print('total words: %d' % len(glove.stoi))
    words = ['chip', 'china', 'england', 'woman', 'king']
    for word in words:
        print("similar word of: ", word)
        get_similar_tokens_knn(word, 3, glove)
    

if __name__ == '__main__':
    #do_train()
    test_pretrained_word2vec()








