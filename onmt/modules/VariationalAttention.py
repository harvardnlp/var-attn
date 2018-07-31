from collections import namedtuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.Utils import aeq, sequence_mask, Params, DistInfo


class VariationalAttention(nn.Module):
    def __init__(
        self, src_dim, tgt_dim,
        attn_dim,
        p_dist_type="categorical",
        q_dist_type="categorical",
        use_prior=False,
        scoresF=F.softplus,
        n_samples=1,
        mode="sample",
        attn_type="mlp",
    ):
        super(VariationalAttention, self).__init__()

        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.attn_dim = attn_dim
        self.p_dist_type = p_dist_type
        self.q_dist_tyqe = q_dist_type
        self.use_prior = use_prior
        self.scoresF = scoresF
        self.n_samples = n_samples
        self.mode = mode
        self.attn_type = attn_type
        self.dim = attn_dim
        dim = self.dim
        self.k = 0

        if self.attn_type == "general":
            self.linear_in = nn.Linear(tgt_dim, src_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(src_dim, dim, bias=False)
            self.linear_query = nn.Linear(tgt_dim, dim, bias=False)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(src_dim + tgt_dim, tgt_dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)

        if self.attn_type == "general":
            h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
            h_t_ = self.linear_in(h_t_)
            h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        elif self.attn_type == "mlp":
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, self.tgt_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, self.src_dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def sample_attn(self, params, n_samples=1, lengths=None, mask=None):
        dist_type = params.dist_type
        if dist_type == "categorical":
            alpha = params.alpha
            log_alpha = params.log_alpha
            K = n_samples
            N = alpha.size(0)
            T = alpha.size(1)
            S = alpha.size(2)
            attns_id = torch.distributions.categorical.Categorical(
               alpha.view(N*T, S)
            ).sample(
                torch.Size([n_samples])
            ).view(K, N, T, 1)
            attns = torch.Tensor(K, N, T, S).zero_().cuda()
            attns.scatter_(3, attns_id, 1)
            attns = attns.to(alpha)
            # log alpha: K, N, T, S
            log_alpha = log_alpha.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs = log_alpha.gather(3, attns_id.to(log_alpha.device)).squeeze(3)
            return attns, sample_log_probs
        else:
            raise Exception("Unsupported dist")
        return attns, None

    def sample_attn_wsram(self, q_scores, p_scores, n_samples=1, lengths=None, mask=None):
        dist_type = q_scores.dist_type
        assert p_scores.dist_type == dist_type
        if dist_type == "categorical":
            alpha_q = q_scores.alpha
            log_alpha_q = q_scores.log_alpha
            K = n_samples
            N = alpha_q.size(0)
            T = alpha_q.size(1)
            S = alpha_q.size(2)
            attns_id = torch.distributions.categorical.Categorical(
               alpha_q.view(N*T, S)
            ).sample(
                torch.Size([n_samples])
            ).view(K, N, T, 1)
            attns = torch.Tensor(K, N, T, S).zero_().cuda()
            attns.scatter_(3, attns_id, 1)
            q_sample = attns.to(alpha_q)
            # log alpha: K, N, T, S
            log_alpha_q = log_alpha_q.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs_q = log_alpha_q.gather(3, attns_id.to(log_alpha_q.device)).squeeze(3)
            log_alpha_p = p_scores.log_alpha
            log_alpha_p = log_alpha_p.unsqueeze(0).expand(K, N, T, S)
            sample_log_probs_p = log_alpha_p.gather(3, attns_id.to(log_alpha_p.device)).squeeze(3)
            sample_p_div_q_log = sample_log_probs_p - sample_log_probs_q
            return q_sample, sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log
        else:
            raise Exception("Unsupported dist")

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None, q_scores=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          q_scores (`FloatTensor`): the attention params from the inference network

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Weighted context vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
          * Unormalized attention scores for each query 
            `[batch x tgt_len x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
            if q_scores is not None:
                # oh, I guess this is super messy
                if q_scores.alpha is not None:
                    q_scores = Params(
                        alpha=q_scores.alpha.unsqueeze(1),
                        log_alpha=q_scores.log_alpha.unsqueeze(1),
                        dist_type=q_scores.dist_type,
                    )
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.
        # Params should be T x N x S
        if self.p_dist_type == "categorical":
            scores = self.score(input, memory_bank)
            if memory_lengths is not None:
                # mask : N x T x S
                mask = sequence_mask(memory_lengths)
                mask = mask.unsqueeze(1)  # Make it broadcastable.
                scores.data.masked_fill_(1 - mask, -float('inf'))
            if self.k > 0 and self.k < scores.size(-1):
                topk, idx = scores.data.topk(self.k)
                new_attn_score = torch.zeros_like(scores.data).fill_(float("-inf"))
                new_attn_score = new_attn_score.scatter_(2, idx, topk)
                scores = new_attn_score
            log_scores = F.log_softmax(scores, dim=-1)
            scores = log_scores.exp()

            c_align_vectors = scores

            p_scores = Params(
                alpha=scores,
                log_alpha=log_scores,
                dist_type=self.p_dist_type,
            )

        # each context vector c_t is the weighted average
        # over all the source hidden states
        context_c = torch.bmm(c_align_vectors, memory_bank)
        if self.mode != 'wsram':
            concat_c = torch.cat([input, context_c], -1)
            # N x T x H
            h_c = self.tanh(self.linear_out(concat_c))
        else:
            h_c = None

        # sample or enumerate
        # y_align_vectors: K x N x T x S
        q_sample, p_sample, sample_log_probs = None, None, None
        sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log = None, None, None
        if self.mode == "sample":
            if q_scores is None or self.use_prior:
                p_sample, sample_log_probs = self.sample_attn(
                    p_scores, n_samples=self.n_samples,
                    lengths=memory_lengths, mask=mask if memory_lengths is not None else None)
                y_align_vectors = p_sample
            else:
                q_sample, sample_log_probs = self.sample_attn(
                    q_scores, n_samples=self.n_samples,
                    lengths=memory_lengths, mask=mask if memory_lengths is not None else None)
                y_align_vectors = q_sample
        elif self.mode == "enum" or self.mode == "exact":
            y_align_vectors = None
        elif self.mode == "wsram":
            assert q_scores is not None
            q_sample, sample_log_probs_q, sample_log_probs_p, sample_p_div_q_log = self.sample_attn_wsram(
                q_scores, p_scores, n_samples=self.n_samples,
                lengths=memory_lengths, mask=mask if memory_lengths is not None else None)
            y_align_vectors = q_sample


        # context_y: K x N x T x H
        if y_align_vectors is not None:
            context_y = torch.bmm(
                y_align_vectors.view(-1, targetL, sourceL),
                memory_bank.unsqueeze(0).repeat(self.n_samples, 1, 1, 1).view(-1, sourceL, dim)
            ).view(self.n_samples, batch, targetL, dim)
        else:
            # For enumerate, K = S.
            # memory_bank: N x S x H
            context_y = (memory_bank
                .unsqueeze(0)
                .repeat(targetL, 1, 1, 1) # T, N, S, H
                .permute(2, 1, 0, 3)) # S, N, T, H
        input = input.unsqueeze(0).repeat(context_y.size(0), 1, 1, 1)
        concat_y = torch.cat([input, context_y], -1)
        # K x N x T x H
        h_y = self.tanh(self.linear_out(concat_y))

        if one_step:
            if h_c is not None:
                # N x H
                h_c = h_c.squeeze(1)
            # N x S
            c_align_vectors = c_align_vectors.squeeze(1)
            context_c = context_c.squeeze(1)

            # K x N x H
            h_y = h_y.squeeze(2)
            # K x N x S
            #y_align_vectors = y_align_vectors.squeeze(2)

            q_scores = Params(
                alpha = q_scores.alpha.squeeze(1) if q_scores.alpha is not None else None,
                dist_type = q_scores.dist_type,
                samples = q_sample.squeeze(2) if q_sample is not None else None,
                sample_log_probs = sample_log_probs.squeeze(2) if sample_log_probs is not None else None,
                sample_log_probs_q = sample_log_probs_q.squeeze(2) if sample_log_probs_q is not None else None,
                sample_log_probs_p = sample_log_probs_p.squeeze(2) if sample_log_probs_p is not None else None,
                sample_p_div_q_log = sample_p_div_q_log.squeeze(2) if sample_p_div_q_log is not None else None,
            ) if q_scores is not None else None
            p_scores = Params(
                alpha = p_scores.alpha.squeeze(1),
                log_alpha = log_scores.squeeze(1),
                dist_type = p_scores.dist_type,
                samples = p_sample.squeeze(2) if p_sample is not None else None,
            )

            if h_c is not None:
                # Check output sizes
                batch_, dim_ = h_c.size()
                aeq(batch, batch_)
                batch_, sourceL_ = c_align_vectors.size()
                aeq(batch, batch_)
                aeq(sourceL, sourceL_)
        else:
            assert False
            # Only support input feeding.
            # T x N x H
            h_c = h_c.transpose(0, 1).contiguous()
            # T x N x S
            c_align_vectors = c_align_vectors.transpose(0, 1).contiguous()

            # T x K x N x H
            h_y = h_y.permute(2, 0, 1, 3).contiguous()
            # T x K x N x S
            #y_align_vectors = y_align_vectors.permute(2, 0, 1, 3).contiguous()

            q_scores = Params(
                alpha = q_scores.alpha.transpose(0, 1).contiguous(),
                dist_type = q_scores.dist_type,
                samples = q_sample.permute(2, 0, 1, 3).contiguous(),
            )
            p_scores = Params(
                alpha = p_scores.alpha.transpose(0, 1).contiguous(),
                log_alpha = log_alpha.transpose(0, 1).contiguous(),
                dist_type = p_scores.dist_type,
                samples = p_sample.permute(2, 0, 1, 3).contiguous(),
            )

            # Check output sizes
            targetL_, batch_, dim_ = h_c.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = c_align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        # For now, don't include samples.
        dist_info = DistInfo(
            q = q_scores,
            p = p_scores,
        )

        # h_y: samples from simplex
        #   either K x N x H, or T x K x N x H
        # h_c: convex combination of memory_bank for input feeding
        #   either N x H, or T x N x H
        # align_vectors: convex coefficients / boltzmann dist
        #   either N x S, or T x N x S
        # raw_scores: unnormalized scores
        #   either N x S, or T x N x S
        return h_y, h_c, context_c, c_align_vectors, dist_info
