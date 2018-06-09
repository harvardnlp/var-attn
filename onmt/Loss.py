"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io

from torch.distributions import Dirichlet as Dir
from torch.distributions.categorical import Categorical as Cat
from torch.distributions.kl import kl_divergence


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns, dist_info=None, output_baseline=None):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        if dist_info is not None:
            assert (dist_info.p.dist_type == dist_info.q.dist_type)
            self.dist_type = dist_info.q.dist_type
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns, dist_info=dist_info, output_baseline=output_baseline)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization, dist_info=None,
                             output_baseline=None):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns, dist_info=dist_info, output_baseline=output_baseline)
        if dist_info is not None:
            assert (dist_info.p.dist_type == dist_info.q.dist_type)
            self.dist_type = dist_info.q.dist_type

        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(normalization).backward(retain_graph=True)
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, xent, kl, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        return onmt.Statistics(xent.item(), kl.item(), non_padding.sum().item(), num_correct.item())

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert (label_smoothing >= 0.0 and label_smoothing <= 1.0)
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing
        self.alpha = 1

    def _make_shard_state(self, batch, output, range_, attns=None,
                          dist_info=None, output_baseline=None):
        state = {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

        # whoops, maybe I need to make everything T first?
        if dist_info.p is not None:
            state["p_samples"] = dist_info.p.samples
            if dist_info.p.dist_type == "dirichlet":
                state["p_alpha"] = dist_info.p.alpha
            elif dist_info.p.dist_type == "categorical":
                state["p_alpha"] = dist_info.p.alpha
            else:
                raise Exception("Unimplemented distribution")
        #import pdb; pdb.set_trace()
        if dist_info.q is not None:
            state["q_samples"] = dist_info.q.samples
            if dist_info.q.dist_type == "dirichlet":
                state["q_alpha"] = dist_info.q.alpha
            elif dist_info.q.dist_type == "categorical":
                state["q_alpha"] = dist_info.q.alpha
                state["q_log_alpha"] = dist_info.q.log_alpha
                state["q_sample_log_probs"] = dist_info.q.sample_log_probs
            else:
                raise Exception("Unimplemented distribution")

        assert output_baseline is not None
        if output_baseline is not None:
            state["output_baseline"] = output_baseline
        return state

    def _compute_loss(
        self, batch, output, target,
        p_samples=None, q_samples=None,
        p_alpha=None, q_alpha=None,
        q_log_alpha=None,
        q_sample_log_probs=None,
        output_baseline=None
    ):
        # Reconstruction
        output_baseline = output_baseline.unsqueeze(1)
        # TODO(jchiu): hacky, just use q for now, but switch on something later.
        scores = self.generator(output, q_log_alpha)
        scores = scores.view(-1, scores.size(-1))
        scores_baseline = self.generator(output_baseline)
        scores_baseline = scores_baseline.view(-1, scores.size(-1))


        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        xent = self.criterion(scores, gtruth)

        if q_sample_log_probs is not None:
            # This code doesn't handle multiple samples
            scores_nopad = scores[gtruth.ne(self.padding_idx)]
            scores_baseline_nopad = scores_baseline[gtruth.ne(self.padding_idx)]
            gtruth_nopad = gtruth[gtruth.ne(self.padding_idx)]
            llh_ind = scores_nopad.gather(1, gtruth_nopad.unsqueeze(1))
            llh_baseline_ind = scores_baseline_nopad.gather(1, gtruth_nopad.unsqueeze(1))
            reward = (llh_ind.detach() - llh_baseline_ind.detach()).view(-1) # T*N
            q_sample_log_probs = q_sample_log_probs.view(-1) # T, N
            q_sample_log_probs = q_sample_log_probs[gtruth.ne(self.padding_idx)]
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            xent_data = xent.data.clone()
        else:
            xent_data = xent.data.clone()

        # KL
        if q_alpha is not None:
            q_alpha = q_alpha.contiguous().view(-1, q_alpha.size(2))
            q_alpha = q_alpha[gtruth.ne(self.padding_idx)]
            p_alpha = p_alpha.contiguous().view(-1, p_alpha.size(2))
            p_alpha = p_alpha[gtruth.ne(self.padding_idx)]
            if self.dist_type == 'dirichlet':
                q = Dir(q_alpha)
                p = Dir(p_alpha)
            elif self.dist_type == 'categorical':
                q = Cat(q_alpha)
                p = Cat(p_alpha)
                #import pdb; pdb.set_trace()
            else:
                assert (False)
            kl = kl_divergence(q, p).sum()
            loss = xent + self.alpha * kl
        else:
            kl = torch.zeros(1).to(xent)
            loss = xent

        # subtract reward
        if q_sample_log_probs is not None:
            loss = loss - (reward * q_sample_log_probs).sum()
        #import pdb; pdb.set_trace()

        kl_data = kl.data.clone()

        stats = self._stats(xent_data, kl_data, scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None and v is not None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                if v_split[0].grad is not None:
                    variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
