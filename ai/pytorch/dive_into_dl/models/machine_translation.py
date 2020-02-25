import collections
import os
import io
import math
import torch
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_on_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) -1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens), 
                        specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
    
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens =in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            print('too long, ignore!', line)
            continue
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)


def test():
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    print('in vocab size, ', len(in_vocab), 'out vocab size', len(out_vocab))
    print(dataset[0])


if __name__ == '__main__':
    test()







