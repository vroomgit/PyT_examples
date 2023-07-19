# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:04:59 2023

@author: varun.singh
"""
# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

device = torch.device("cpu")

# Data.py
import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            #c=0
            for line in f:
                #c+=1
                words = line.split() + ['<eos>']
                #print("Serial ",c,words)
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
                
            ids = torch.cat(idss)
            #print(ids)
        return ids
    
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz # For Train: 2088628/256 = 8158.703125
    print("nbatch size: ", nbatch)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz) # drop remainders - 8158
    print("data size: ", data.shape)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous() # transpose 
    #print("batchify final shape: ",data.shape)
    return data.to(device) # For train data, with a batch size of 256, output is [8158, 256]

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    print("get_batch") 
    #print(seq_len) # if seq_len (35) is less than 8158 (for train) then this is 35
    data = source[i:i+seq_len]
    print("X shape", data.shape) # [35, 256]
    decoder_tgt = source[i+1:i+1+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    print("Y shape", target.shape) # [8960] (35 * 256) 
    return data, target, decoder_tgt

k = Corpus("./data/wikitext-2/")

train_data = batchify(k.train, 128)    

b1, t1, dt1 = get_batch(train_data, 0)    

print(len(k.dictionary))

em = nn.Embedding(33278,256)

emb = em(b1)
emb_dec = em(dt1)

t = nn.Transformer(d_model=256)

tr_out = t(emb, emb_dec)

#_______________

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


src = emb
tgt = emb_dec
src_mask = _generate_square_subsequent_mask(emb.shape[0])
tgt_mask = _generate_square_subsequent_mask(emb_dec.shape[0])

output = t(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)



