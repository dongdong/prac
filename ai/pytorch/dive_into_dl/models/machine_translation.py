import collections
import os
import io
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
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


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0,
            **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)
    
    def forward(self, inputs, state):
        # input: (batch, seq_len)
        # embedding: (batch, seq_len, embed_size)
        # permute: (seq_len, batch, embed_size)
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        # rnn output: (seq_len, batch, num_hidden)
        # rnn state: (num_layers, batch, num_hidden)
        return self.rnn(embedding, state)
    
    def begin_state(self):
        return None
    

def attention_model(input_size, attention_size):
    model = nn.Sequential(
            # (seq_len, batch, input_size)
            nn.Linear(input_size, attention_size, bias=False),
            # (seq_len, batch, attention_size)
            nn.Tanh(),
            # (seq_len, batch, 1)
            nn.Linear(attention_size, 1, bias=False))
    return model


def attention_forward(model, enc_states, dec_state):
    '''
    enc_states: (seq_len, batch, num_hidden)
    dec_state: (batch, num_hidden)
    a(s, h) = v.T * tanh(W_s * s + W_h * h)
    '''
    # dec_states: (seq_len, batch, num_hidden)
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    # enc_and_dec_states: (seq_len, batch, num_hidden*2)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    # e: (seq_len, batch, 1)
    e = model(enc_and_dec_states)
    # alpha: (seq_len, batch, 1)
    alpha = F.softmax(e, dim=0)
    # (seq_len, batch, 1) * (seq_len, batch, num_hidden)
    # (batch, num_hidden)
    return (alpha * enc_states).sum(dim=0)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, 
            attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2*num_hiddens, attention_size)
        self.rnn = nn.GRU(2*embed_size, num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)
    
    def forward(self, cur_input, state, enc_states):
        '''
        cur_input: (batch, )
        state: (num_layers, batch, num_hiddens)
        '''        
        # c: (batch, num_hiddens)
        c = attention_forward(self.attention, enc_states, state[-1])
        # input_and_c: (batch, embed_size + num_hidden) == (batch_size, embed_size*2)
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        # output: (1, batch, num_hidden)
        # state: (num_layers, batch, num_hidden)
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # output: (batch, num_hidden)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        return enc_state


def batch_loss(encoder, decoder, X, Y, loss, out_vocab):
    batch_size = X.shape[0]
    # enc_state: (num_layers, batch, hidden_size)
    enc_state = encoder.begin_state()
    # enc_outputs: (seq_len, batch, hidden_size)
    # enc_state: (num_layers, batch, hidden_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # dec_state == enc_state
    dec_state = decoder.begin_state(enc_state)
    # dec_input = (batch, )
    dec_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)
    mask, num_not_pad_tokens = torch.ones(batch_size, ), 0
    l = torch.tensor([0.0])
    # Y: (batch, seq_len) => (seq_len, batch)
    # y: (batch, )
    for y in Y.permute(1, 0):
        # dec_output: (batch, num_hidden)
        # dec_state: (num_layers, batch, num_hidden)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l += (mask * loss(dec_output, y)).sum()
        dec_input = y
        num_not_pad_tokens += mask.sum().item()
        mask = mask * (y != out_vocab.stoi[EOS]).float()
    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss, out_vocab)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


def translate(encoder, decoder, input_seq, max_seq_len, in_vocab, out_vocab):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == EOS:
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


def do_train():
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    embed_size, num_hiddens, num_layers = 64, 64, 2
    attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
    encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, 
            drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
            attention_size, drop_prob)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab)
    input_seq = 'ils regardent .'
    output_seq = translate(encoder, decoder, input_seq, max_seq_len, in_vocab, 
            out_vocab)
    print(input_seq, "==>", output_seq)



def test():
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    print('in vocab size', len(in_vocab), 'out vocab size', len(out_vocab))
    print(dataset[0])

    encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    output, state = encoder(torch.zeros((4, 7)), encoder.begin_state())
    # (4, 7) -> (4, 7, 8) -> (7, 4, 8) -> (7, 4, 16), (2, 4, 16) 
    print(output.shape, state.shape)

    seq_len, batch_size, num_hiddens = 10, 4, 8
    model = attention_model(2*num_hiddens, 10)
    enc_states = torch.zeros((seq_len, batch_size, num_hiddens))
    dec_state = torch.zeros((batch_size, num_hiddens))  
    ret = attention_forward(model, enc_states, dec_state)  
    print(ret.shape)



if __name__ == '__main__':
    #test()
    do_train()


