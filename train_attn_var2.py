#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import h5py
import time
import logging
import pickle

from data import Dataset
from models_attn_var2 import AttnNetwork

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='/n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-train.hdf5')
parser.add_argument('--val_file', default='/n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-val.hdf5')
parser.add_argument('--src_vocab', default='/n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe.src.dict')
parser.add_argument('--tgt_vocab', default='/n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe.targ.dict')

parser.add_argument('--vocab_file', default='')
parser.add_argument('--train_from', default='')

# Model options
parser.add_argument('--word_dim', default=512, type=int)
parser.add_argument('--h_dim', default=512, type=int)
parser.add_argument('--dec_h_dim', default=768, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--attn', default='vae', type=str)
parser.add_argument('--mode', default='train', type=str)

# Optimization options
parser.add_argument('--param_init', default=0.1, type=float)
parser.add_argument('--checkpoint_path', default='baseline.pt')
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--min_epochs', default=7, type=int)
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=2, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=1000)


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)  
  train_sents = train_data.batch_size.sum()
  src_vocab = int(train_data.source_vocab)
  tgt_vocab = int(train_data.target_vocab)
  
  print('Train data: %d batches' % len(train_data))
  print('Val data: %d batches' % len(val_data))
  print('Source vocab size: %d' % src_vocab)
  print('Target vocab size: %d' % tgt_vocab)
  if args.slurm == 0:
    cuda.set_device(args.gpu)

  num_params = 0
  if args.train_from == '':
    model = AttnNetwork(src_vocab = src_vocab,
                        tgt_vocab = tgt_vocab,                        
                        word_dim = args.word_dim,
                        h_dim = args.h_dim,
                        dec_h_dim = args.dec_h_dim,
                        num_layers = args.num_layers,
                        dropout = args.dropout,
                        mode = args.attn)
    for param in model.parameters():    
      param.data.uniform_(-args.param_init, args.param_init)
      num_params += param.view(-1).size()[0]
  else:
    print('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
  print("model architecture")
  print(model)
  print("params", num_params)
  if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  elif args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
  model.train()
  if args.gpu >= 0:
    model.cuda()

    
  best_val_ppl = 1e5
  # val_ppl  = eval(val_data, model)

  if args.mode == 'test':
    model.eval()
    # save_attn(val_data, model)
    model.mode = args.attn
    # exit()
    model.mode = 'soft'
    soft_ppl = eval(val_data, model)
    model.mode = 'hard'
    hard_ppl = eval(val_data, model)
    model.mode = 'vae'
    vae_ppl = eval(val_data, model)
    model.mode = 'vae_prior'
    vae_prior_ppl = eval(val_data, model)
    kmax_ppl = []
    print_kmax_str = []
    for k in np.arange(1, 6):
      model.k = int(k)
      model.mode = 'kmax'        
      kmax_ppl.append(eval(val_data, model, False))
      print_kmax_str.append(str(k)+'-Max: %.2f')      
    print_kmax_str = ", ".join(print_kmax_str)
    print('----------Perplexity-------------')
    print('SoftPPL: %.2f, HardPPL: %.2f, VAEPPL: %.2f, VAEPRIORPPL: %.2f' %
          (soft_ppl, hard_ppl, vae_ppl, vae_prior_ppl))
    print(print_kmax_str % tuple(kmax_ppl))
    assert(1==0)
    
  epoch = 0
  num_steps = 0
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_bound = 0.
    train_kl = 0.
    num_sents = 0
    num_words = 0
    b = 0
    
    for i in np.random.permutation(len(train_data)):
      src, tgt, tgt_mask, src_l, tgt_l, batch_size, nonzeros = train_data[i]
      if args.gpu >= 0:
        src, tgt, tgt_mask = src.cuda(), tgt.cuda(), tgt_mask.cuda()
      b += 1
      num_steps += 1
      optimizer.zero_grad()
      nll, kl = model(src, tgt, tgt_mask)
      (nll + kl).backward()
      if args.attn == 'vae_sample' or args.attn == 'vae_sample_prior':
        nll = -model.log_prob
      train_nll += nll.item()*batch_size.item()
      train_kl += kl.item()*batch_size.item()
      train_bound += (nll.item() + kl.item())*batch_size        
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)        
      optimizer.step()
      num_sents += batch_size.item()
      num_words += nonzeros.item()
      
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        print('Epoch: %d, Batch: %d/%d, LR: %.4f, TrainPPL: %.2f, TrainReconPPL: %.2f, TrainKL: %.4f, Optim: %s, |Param|: %.4f, BestValPerf: %.2f, Throughput: %.2f examples/sec' % 
              (epoch, b, len(train_data), args.lr, np.exp(train_bound/num_words),
               np.exp(train_nll / num_words), train_kl / num_words, args.optim,
               param_norm, best_val_ppl, num_sents / (time.time() - start_time)))            
    print('Checking validation perf...')
    val_ppl  = eval(val_data, model)
    if val_ppl < best_val_ppl:
      # model.mode = 'soft'
      # soft_ppl = eval(val_data, model, False)
      model.mode = 'hard'
      hard_ppl = eval(val_data, model, False)
      print('hard', hard_ppl)
      # model.mode = 'vae'
      # vae_ppl = eval(val_data, model, False)            
      # kmax_ppl = []
      # print_kmax_str = []
      # for k in np.arange(1, 6):
      #   model.k = int(k)
      #   model.mode = 'kmax'        
      #   kmax_ppl.append(eval(val_data, model, False))
      #   # kmax_acc.append(eval_argmax(val_data, model, False))
      #   print_kmax_str.append(str(k)+'-Max: %.2f')
      # print_kmax_str = ", ".join(print_kmax_str)      
      model.mode = args.attn
      # print('----------Perplexity-------------')
      # print('SoftPPL: %.2f, HardPPL: %.2f, VAEPPL: %.2f' %
      #       (soft_ppl, hard_ppl, vae_ppl))
      # print(print_kmax_str % tuple(kmax_ppl))
      # print('----------Accuracy-------------')
      # print('SoftAcc: %.2f, HardAcc: %.2f' % (soft_acc, hard_acc))
      # print(print_kmax_str % tuple(kmax_acc))
      print('---------------------------------')        
      best_val_ppl = val_ppl
      checkpoint = {
        'args': args.__dict__,
        'model': model,
        'optimizer': optimizer
      }
      print('Saving checkpoint to %s' % args.checkpoint_path)
      torch.save(checkpoint, args.checkpoint_path)
    else:
      if epoch >= args.min_epochs:
        args.lr = args.lr*0.5      
        for param_group in optimizer.param_groups:
          param_group['lr'] = args.lr

def eval_argmax(data, model, print_result = True):
  model.eval()
  num_words = 0
  total_correct = 0.
  for i in range(len(data)):  
    src, tgt, tgt_mask, src_l, tgt_l, batch_size, nonzeros = data[i]
    if args.gpu >= 0:
      src, tgt, tgt_mask = src.cuda(), tgt.cuda(), tgt_mask.cuda()
    src_emb = model.enc_emb(src)
    enc_h, _ = model.enc_rnn(src_emb)
    h = None
    c = None
    gen = [tgt[:, 0]]
    for l in range(tgt_l-1):
      log_prob, h, c = model.forward_one_step(enc_h, gen[-1], h, c)
      next_token = torch.max(log_prob, 1)[1]
      gen.append(next_token.detach())
    tgt_argmax = torch.stack(gen[1:], 1)
    correct = ((tgt_argmax.data == tgt[:, 1:].data).float()*tgt_mask.data).sum()
    total_correct += correct
    num_words += nonzeros
  acc = total_correct/num_words*100
  if print_result:
    print("Acc: %.2f" % acc)
  return acc
  
def eval(data, model, print_result = True):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_bound = 0.
  total_kl = 0.
  total_ent = 0.
  max_prob1 = 0.
  max_prob2 = 0.
  min_prob1 = 0.
  min_prob2 = 0.
  for i in range(len(data)):
    src, tgt, tgt_mask, src_l, tgt_l, batch_size, nonzeros = data[i]
    if args.gpu >= 0:
      src, tgt, tgt_mask = src.cuda(), tgt.cuda(), tgt_mask.cuda()      
    nll, kl = model(src, tgt, tgt_mask)
    if model.mode == 'vae_sample' or model.mode == 'vae_sample_prior':
      nll = -model.log_prob    
    total_ent -= ((model.attn_prob * (model.attn_prob + 1e-6).log()).sum(2)*tgt_mask).sum().item()
    kmax_attn = model.attn_prob.topk(2)[0]
    max_prob1 += (kmax_attn[:, :, 0]*tgt_mask).sum().item()
    max_prob2 += (kmax_attn[:, :, 1]*tgt_mask).sum().item()
    kmin_attn = (-model.attn_prob).topk(2)[0]
    min_prob1 -= (kmin_attn[:, :, 0]*tgt_mask).sum().item()
    min_prob2 -= (kmin_attn[:, :, 1]*tgt_mask).sum().item()    
    total_nll += nll.item()*batch_size.item()
    total_bound += (nll.item() + kl.item())*batch_size.item()
    total_kl += kl.item()*batch_size.item()
    num_sents += batch_size.item()
    num_words += nonzeros.item()
  recon_ppl = np.exp(total_nll / num_words)
  ppl = np.exp(total_bound / num_words)
  kl = total_kl / num_words
  entropy = total_ent / num_words
  max_prob1 = max_prob1 / num_words
  max_prob2 = max_prob2 / num_words
  min_prob1 = min_prob1 / num_words
  min_prob2 = min_prob2 / num_words  
  if print_result:
    print('PPL: %.4f, ReconPPL: %.4f, KL: %.4f, Entropy: %.4f, AvgMax1: %.4f, AvgMax2: %.4f, AvgMin1: %.4f, AgvMin2: %.4f' %
          (ppl, recon_ppl, kl, entropy, max_prob1, max_prob2, min_prob1, min_prob2))
  model.train()
  return ppl

def save_attn(data, model, print_result = True):
  model.eval()
  src_idx2vocab = {}
  src_vocab2idx = {}  
  with open(args.src_vocab, 'r') as f:
    for line in f:
      word, idx = line.strip().split()
      src_idx2vocab[int(idx.strip())] = word.strip()
      src_vocab2idx[word.strip()] = int(idx.strip())
  tgt_idx2vocab = {}
  tgt_vocab2idx = {}
  attn_data = [] #src, pred, gold, attn
  with open(args.tgt_vocab, 'r') as f:
    for line in f:
      word, idx = line.strip().split()
      tgt_idx2vocab[int(idx.strip())] = word.strip()
      tgt_vocab2idx[word.strip()] = int(idx.strip())
  model.mode = args.attn
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_bound = 0.
  total_kl = 0.
  total_ent = 0.
  max_prob1 = 0.
  max_prob2 = 0.
  min_prob1 = 0.
  min_prob2 = 0.
  p_attn = []
  q_attn = []
  src_str = []
  tgt_str = []
  for i in range(len(data)):
    src, tgt, tgt_mask, src_l, tgt_l, batch_size, nonzeros = data[i]
    if args.gpu >= 0:
      src, tgt, tgt_mask = src.cuda(), tgt.cuda(), tgt_mask.cuda()      
    nll, kl = model(src, tgt, tgt_mask)
    if model.mode == 'vae_sample' or model.mode == 'vae_sample_prior':
      nll = -model.log_prob    
    total_ent -= ((model.attn_prob * (model.attn_prob + 1e-6).log()).sum(2)*tgt_mask).sum().item()
    kmax_attn = model.attn_prob.topk(2)[0]
    max_prob1 += (kmax_attn[:, :, 0]*tgt_mask).sum().item()
    max_prob2 += (kmax_attn[:, :, 1]*tgt_mask).sum().item()
    kmin_attn = (-model.attn_prob).topk(2)[0]
    min_prob1 -= (kmin_attn[:, :, 0]*tgt_mask).sum().item()
    min_prob2 -= (kmin_attn[:, :, 1]*tgt_mask).sum().item()    
    total_nll += nll.item()*batch_size.item()
    total_bound += (nll.item() + kl.item())*batch_size.item()
    total_kl += kl.item()*batch_size.item()
    num_sents += batch_size.item()
    num_words += nonzeros.item()
    for b in range(batch_size):
      src_b = " ".join([src_idx2vocab[idx] for idx in list(src[b].data)])
      tgt_b_l = int(tgt_mask[b].data.sum())
      tgt_b = " ".join([tgt_idx2vocab[idx] for idx in list(tgt[b][1:1+tgt_b_l].data)])
      src_str.append(src_b)
      tgt_str.append(tgt_b)
      p_attn.append(model.attn_prob[b][:tgt_b_l].data.cpu().numpy())
      q_attn.append(model.q_attn_prob[b][:tgt_b_l].data.cpu().numpy())
    # print(src_str[-1])
    # print(tgt_str[-1])
    # print(p_attn[-1])
    # print(q_attn[-1])
    print(len(src_str[-1].split()), len(tgt_str[-1].split()), p_attn[-1].shape, q_attn[-1].shape)
  attn_data = [src_str, tgt_str, p_attn, q_attn]
  print(len(attn_data), len(attn_data[0]), len(attn_data[1]), len(attn_data[2]), len(attn_data[3]))
  pickle.dump(attn_data, open(args.checkpoint_path + '.pkl', 'wb'))      
  recon_ppl = np.exp(total_nll / num_words)
  ppl = np.exp(total_bound / num_words)
  kl = total_kl / num_words
  entropy = total_ent / num_words
  max_prob1 = max_prob1 / num_words
  max_prob2 = max_prob2 / num_words
  min_prob1 = min_prob1 / num_words
  min_prob2 = min_prob2 / num_words  
  if print_result:
    print('PPL: %.4f, ReconPPL: %.4f, KL: %.4f, Entropy: %.4f, AvgMax1: %.4f, AvgMax2: %.4f, AvgMin1: %.4f, AgvMin2: %.4f' %
          (ppl, recon_ppl, kl, entropy, max_prob1, max_prob2, min_prob1, min_prob2))
  model.train()
  return ppl

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
