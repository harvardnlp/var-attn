#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
#import Concrete

class AttnNetwork(nn.Module):
  def __init__(self, src_vocab=10000,
               tgt_vocab=1000,
               word_dim=300,
               h_dim=300,
               dec_h_dim = 300,
               num_layers=2,
               dropout=0,
               mode='soft',
               gamma = 0):
    super(AttnNetwork, self).__init__()
    self.num_layers = num_layers
    self.gamma = gamma
    self.enc_emb = nn.Embedding(src_vocab, word_dim)    
    self.enc_rnn = nn.LSTM(word_dim, h_dim, num_layers = num_layers,
                           dropout = dropout, batch_first = True, bidirectional=True)
    self.dec_emb = nn.Embedding(tgt_vocab, word_dim)
    self.dec_rnn = nn.LSTM(word_dim + h_dim*2, dec_h_dim, num_layers = num_layers,
                           dropout = dropout, batch_first = True)
    self.context_proj = nn.Sequential(nn.Linear(h_dim*2 + dec_h_dim, dec_h_dim),
                                    nn.Tanh())
    self.vocab_proj = nn.Linear(dec_h_dim, tgt_vocab)
    self.dropout = nn.Dropout(dropout)
    self.attn_proj = nn.Linear(h_dim*2, h_dim, bias=False)
    self.attn_proj2 = nn.Linear(dec_h_dim, h_dim, bias = False)
    self.attn_proj3 = nn.Linear(h_dim, 1, bias = False)
    self.dec_h_dim = dec_h_dim
    self.h_dim = h_dim
    self.mode = mode
    if 'vae' in mode:
      self.q_dim = h_dim
      self.q_src = nn.LSTM(word_dim, self.q_dim, num_layers = num_layers, bidirectional=True,
                           dropout = dropout, batch_first = True)
      self.q_tgt = nn.LSTM(word_dim, self.q_dim, num_layers = num_layers, bidirectional=True,
                           dropout = dropout, batch_first = True)
      self.q_proj = nn.Linear(self.q_dim*2, self.q_dim*2, bias = False)
    self.zero = Variable(torch.zeros(1).cuda())
    self.dropout = nn.Dropout(dropout)
    self.k = 0
    
  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d
    
  def forward_one_step(self, enc_h, token, h = None, prev_context = None):
    b = enc_h.size(0)
    src_l  = enc_h.size(1)
    tgt_emb = self.dec_emb(token).unsqueeze(1) #b x 1 x word_dim
    if h is None:
      h0 = Variable(torch.zeros(self.num_layers, b, self.dec_h_dim).type_as(enc_h.data))
      c0 = Variable(torch.zeros(self.num_layers, b, self.dec_h_dim).type_as(enc_h.data))
      h = (h0, c0)
      prev_context = Variable(torch.zeros(b, 1, self.h_dim*2).type_as(enc_h.data))
    else:
      h = (h[0].detach(), h[1].detach())
      prev_context = prev_context.detach()
    dec_h, h = self.dec_rnn(torch.cat([tgt_emb, prev_context], 2), h) #(b x 1 x h_dim)
    attn_score = self.attn_proj(enc_h.detach()) # b x src x h_dim
    
    dec_attn_score_i = self.attn_proj2(dec_h) # b x 1 x h_dim
    attn_score = dec_attn_score_i + attn_score # b x src x h_dim
    attn_score = self.attn_proj3(F.tanh(attn_score)).transpose(1,2) # b x 1 x src
    
    # attn_score = torch.matmul(dec_h.detach(), attn_score.transpose(1,2)) # b x 1 x src
    if self.mode == 'kmax' or self.mode == 'kmax_max':
      if src_l > self.k:
        topk, idx = attn_score.data.topk(self.k)
        new_attn_score = torch.zeros_like(attn_score.data).fill_(-1e8)
        new_attn_score = new_attn_score.scatter_(2, idx, topk)
        attn_score = Variable(new_attn_score)      
    attn_log_prob = F.log_softmax(attn_score, 2).detach() # b x 1 x src
    attn_prob = attn_log_prob.exp().detach()
    self.attn_prob = attn_prob
    context = torch.matmul(attn_prob, enc_h) # b x 1 x h    
    if self.mode == 'soft':
      combined = torch.cat([dec_h, context], 2) # b x 1 x 3h
      vocab_score = self.vocab_proj(self.dropout(self.context_proj(combined))) # b x 1 x vocab
      log_prob = F.log_softmax(vocab_score, 2)
      log_prob = log_prob.squeeze(1)
      return log_prob.detach(), h, context
    elif self.mode == 'hard' or self.mode == 'kmax':
      dec_h_i = dec_h.expand(b, src_l, self.dec_h_dim)  # b x src x h
      context_i = torch.cat([dec_h_i, enc_h], 2) # b x src X 3h
      log_prob_i = F.log_softmax(self.vocab_proj(self.dropout(self.context_proj(context_i))), 2).detach()
      attn_i = attn_log_prob.squeeze(1).unsqueeze(2).expand_as(log_prob_i).detach() # b x src x vocab
      log_prob = self.logsumexp(attn_i + log_prob_i, 1) # b x vocab
      return log_prob.detach(), h, context
    elif self.mode == 'hard_max' or self.mode == 'kmax_max':
      dec_h_i = dec_h.expand(b, src_l, self.dec_h_dim)  # b x src x h
      context_i = torch.cat([dec_h_i, enc_h], 2) # b x src X 3h
      log_prob_i = F.log_softmax(self.vocab_proj(self.dropout(self.context_proj(context_i))), 2).detach()
      attn_i = attn_log_prob.squeeze(1).unsqueeze(2).expand_as(log_prob_i).detach() # b x src x vocab
      log_prob = torch.max(attn_i + log_prob_i, 1)[0] # b x vocab
      return log_prob.detach(), h, context

  def forward(self, src, tgt, tgt_mask):
    b = tgt.size(0)
    src_l  = src.size(1)
    tgt_l = tgt.size(1) - 1
    src_emb = self.dropout(self.enc_emb(src))
    enc_h, _ = self.enc_rnn(src_emb) # b x src x 2h
    # DBG
    self.src_emb_h = src_emb
    self.enc_h = enc_h
    tgt_emb = self.dropout(self.dec_emb(tgt)) # b x tgt+1 x h
    p_attn_score = self.attn_proj(enc_h) # b x src x h

    if self.mode == 'vae' or self.mode == 'vae_sample':
      q_src, _ = self.q_src(src_emb) 
      q_tgt, _ = self.q_tgt(tgt_emb[:, 1:])
      q_attn_score = torch.matmul(self.q_proj(q_tgt), q_src.transpose(1,2)) # b x tgt x src      
      q_attn_log_prob = F.log_softmax(q_attn_score, 2)
      q_attn_prob = q_attn_log_prob.exp()      
      q_attn_sample = torch.multinomial(q_attn_prob.view(b*tgt_l, src_l).detach(), 1)
      q_attn_log_prob_score = torch.gather(q_attn_log_prob.view(b*tgt_l, src_l), 1, q_attn_sample)
      q_attn_log_prob_score = q_attn_log_prob_score.view(b, tgt_l)
      q_attn_sample = q_attn_sample.view(b, tgt_l, 1).expand(b, tgt_l, self.h_dim*2).unsqueeze(2)
      
    h = None
    context_i = Variable(torch.zeros(b, 1, self.h_dim*2).cuda())
    p_attn_log_prob = []
    log_prob = []
    log_prob_baseline = []
    prior_log_prob_score = []
    for i in range(tgt_l):
      tgt_emb_i = tgt_emb[:, i].unsqueeze(1) # b x 1 x h
      dec_input = torch.cat([tgt_emb_i, context_i], 2) # b x 1 x 3h
      dec_h_i, h = self.dec_rnn(dec_input, h)
      dec_attn_score_i = self.attn_proj2(dec_h_i) # b x 1 x h_dim
      p_attn_score_i = dec_attn_score_i + p_attn_score # b x src x h_dim
      p_attn_score_i = self.attn_proj3(F.tanh(p_attn_score_i)).transpose(1,2)
      if self.mode == 'kmax' and src_l > self.k:
        topk, idx = p_attn_score_i.data.topk(self.k)
        new_attn_score = torch.zeros_like(p_attn_score_i.data).fill_(-1e8)
        new_attn_score = new_attn_score.scatter_(2, idx, topk)
        p_attn_score_i = Variable(new_attn_score)            
      p_attn_log_prob_i = F.log_softmax(p_attn_score_i, 2)
      p_attn_prob_i = p_attn_log_prob_i.exp() # b x 1 x src
      if self.mode ==  'vae_sample_prior':
        q_attn_sample = torch.multinomial(p_attn_prob_i.view(b, src_l).detach(), 1)
        q_attn_log_prob_score = torch.gather(p_attn_log_prob_i.view(b, src_l), 1, q_attn_sample)
        q_attn_sample = q_attn_sample.unsqueeze(2).expand(b, 1, self.h_dim*2)
        prior_log_prob_score.append(q_attn_log_prob_score)
      context_i = torch.matmul(p_attn_prob_i, enc_h) # b x 1 x h
      if self.mode == 'soft':
        combined = torch.cat([dec_h_i, context_i], 2).squeeze(1)
        vocab_score = self.vocab_proj(self.dropout(self.context_proj(combined)))
        log_prob_i = F.log_softmax(vocab_score, 1) # b x vocab
        log_prob_i = torch.gather(log_prob_i, 1, tgt[:, i+1].unsqueeze(1)).squeeze(1)
      elif self.mode == 'vae_sample_prior':        
        enc_h_i = torch.gather(enc_h, 1, q_attn_sample).squeeze(1) # b x h
        dec_h_i = dec_h_i.squeeze(1)
        q_context_i = torch.cat([dec_h_i, enc_h_i], 1) # b x 3h
        log_prob_i = F.log_softmax(self.vocab_proj(self.dropout(
          self.context_proj(q_context_i))), 1) # b x vocab
        log_prob_i = torch.gather(log_prob_i, 1, tgt[:, i+1].unsqueeze(1)).squeeze(1) # b
        combined = torch.cat([dec_h_i, context_i.squeeze(1)], 1)
        vocab_score = self.vocab_proj(self.dropout(self.context_proj(combined)))
        log_prob_soft_i = F.log_softmax(vocab_score, 1) # b x vocab
        log_prob_soft_i = torch.gather(log_prob_soft_i, 1, tgt[:, i+1].unsqueeze(1)).squeeze(1)
        log_prob_baseline.append(log_prob_soft_i)
      elif self.mode == 'vae_sample':        
        enc_h_i = torch.gather(enc_h, 1, q_attn_sample[:, i]).squeeze(1) # b x h
        dec_h_i = dec_h_i.squeeze(1)
        q_context_i = torch.cat([dec_h_i, enc_h_i], 1) # b x 3h
        log_prob_i = F.log_softmax(self.vocab_proj(self.dropout(
          self.context_proj(q_context_i))), 1) # b x vocab
        log_prob_i = torch.gather(log_prob_i, 1, tgt[:, i+1].unsqueeze(1)).squeeze(1) # b
#        context_i2 = torch.matmul(q_attn_prob[:, i].unsqueeze(1), enc_h)
        combined = torch.cat([dec_h_i, context_i.squeeze(1)], 1)
        vocab_score = self.vocab_proj(self.dropout(self.context_proj(combined)))
        log_prob_soft_i = F.log_softmax(vocab_score, 1) # b x vocab
        log_prob_soft_i = torch.gather(log_prob_soft_i, 1, tgt[:, i+1].unsqueeze(1)).squeeze(1)
        log_prob_baseline.append(log_prob_soft_i)
      else:
        dec_h_i = dec_h_i.expand(b, src_l, self.dec_h_dim)  # b x src x h      
        context = torch.cat([dec_h_i, enc_h], 2) # b x src x 3h
        log_prob_i = F.log_softmax(self.vocab_proj(self.dropout(self.context_proj(
          context))), 2) # b x src x vocab
        log_prob_i = torch.gather(log_prob_i, 2, tgt[:, i+1].unsqueeze(1).expand(
          b, src_l).unsqueeze(2)).squeeze(2) # b x src      
        if self.mode == 'hard' or self.mode == 'kmax':
          log_prob_i = self.logsumexp(p_attn_log_prob_i.squeeze(1) + log_prob_i, 1) # b          
        else:
          if self.mode == 'vae_prior':
            log_prob_i = torch.sum(p_attn_prob_i.squeeze(1)*log_prob_i, 1)
          else:
            log_prob_i = torch.sum(q_attn_prob[:, i]*log_prob_i, 1) # b
      log_prob.append(log_prob_i)
      p_attn_log_prob.append(p_attn_log_prob_i)      
    self.log_prob = torch.stack(log_prob, 1)
    log_prob = (self.log_prob*tgt_mask).sum(1)
    p_attn_log_prob = torch.cat(p_attn_log_prob, 1)
    if self.mode == 'vae_sample_prior':
      q_attn_log_prob_score = torch.cat(prior_log_prob_score, 1)
    if self.mode == 'vae' or self.mode == 'vae_sample':
      kl = q_attn_prob*(q_attn_log_prob-p_attn_log_prob)
      kl = (kl.sum(2)*tgt_mask).sum(1)
    else:# self.mode == 'hard' or self.mode == 'soft' or self.mode == 'kmax':
      kl = self.zero
    self.attn_prob = p_attn_log_prob.detach().exp()
    if self.mode == 'soft' or self.mode == 'hard':
      self.q_attn_prob = self.attn_prob
    else:
      self.q_attn_prob = q_attn_prob
    if self.mode == 'vae_sample' or self.mode == 'vae_sample_prior': 
      log_prob_soft = torch.stack(log_prob_baseline, 1)
      reinforce = (self.log_prob.detach()-log_prob_soft.detach())*q_attn_log_prob_score        
      reinforce = (reinforce*tgt_mask).sum(1)
      self.log_prob = log_prob.mean()
      log_prob_soft = (log_prob_soft*tgt_mask).sum(1).mean()
      return -self.log_prob - reinforce.mean() - log_prob_soft, kl.mean()
    else:                             
      return -log_prob.mean(), kl.mean()
    
