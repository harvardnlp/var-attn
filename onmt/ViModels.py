from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq, sequence_mask, Params, DistInfo
from onmt.Models import MeanEncoder, RNNEncoder, InputFeedRNNDecoder, NMTModel, RNNDecoderState


class InferenceNetwork(nn.Module):
    def __init__(self, inference_network_type, src_embeddings, tgt_embeddings,
                 rnn_type, src_layers, tgt_layers, rnn_size, dropout,
                 attn_type="general",
                 dist_type="none", scoresF=F.softplus):
        super(InferenceNetwork, self).__init__()

        self.inference_network_type = inference_network_type
        self.attn_type = attn_type
        self.dist_type = dist_type

        self.scoresF = scoresF

        if dist_type == "none":
            self.mask_val = float("-inf")
        elif dist_type == "categorical":
            self.mask_val = -float('inf')
        else:
            raise Exception("Invalid distribution type")

        if inference_network_type == 'embedding_only':
            self.src_encoder = MeanEncoder(src_layers, src_embeddings)
            self.tgt_encoder = MeanEncoder(tgt_layers, tgt_embeddings)
        elif inference_network_type == 'brnn':
            self.src_encoder = RNNEncoder(rnn_type, True, src_layers, rnn_size,
                                          rnn_size,
                                          dropout, src_embeddings, False) 
            self.tgt_encoder = RNNEncoder(rnn_type, True, tgt_layers, rnn_size,
                                          rnn_size,
                                          dropout, tgt_embeddings, False) 
        elif inference_network_type == 'bigbrnn':
            self.src_encoder = RNNEncoder(rnn_type, True, src_layers, 2*rnn_size,
                                             2*rnn_size,
                                          dropout, src_embeddings, False) 
            self.tgt_encoder = RNNEncoder(rnn_type, True, tgt_layers, 2*rnn_size,
                                             2*rnn_size,
                                          dropout, tgt_embeddings, False) 
        elif inference_network_type == 'rnn':
            self.src_encoder = RNNEncoder(rnn_type, True, src_layers, rnn_size,
                                          dropout, src_embeddings, False) 
            self.tgt_encoder = RNNEncoder(rnn_type, False, tgt_layers, rnn_size,
                                          dropout, tgt_embeddings, False) 
        if inference_network_type == "bigbrnn":
            self.W = torch.nn.Linear(rnn_size * 2, rnn_size * 2, bias=False)
        else:
            self.W = torch.nn.Linear(rnn_size, rnn_size, bias=False)
        self.rnn_size = rnn_size

    def forward(self, src, tgt, src_lengths=None, src_emb=None, tgt_emb=None):
        src_final, src_memory_bank = self.src_encoder(src, src_lengths, emb=src_emb)
        src_length, batch_size, rnn_size = src_memory_bank.size()

        tgt_final, tgt_memory_bank = self.tgt_encoder(tgt, emb=tgt_emb)
        self.q_src_h = src_memory_bank
        self.q_tgt_h = tgt_memory_bank

        src_memory_bank = src_memory_bank.transpose(0,1) # batch_size, src_length, rnn_size
        src_memory_bank = src_memory_bank.transpose(1,2) # batch_size, rnn_size, src_length
        tgt_memory_bank = self.W(tgt_memory_bank.transpose(0,1)) # batch_size, tgt_length, rnn_size

        if self.dist_type == "categorical":
            scores = torch.bmm(tgt_memory_bank, src_memory_bank)
            # mask source attention
            assert (self.mask_val == -float('inf'))
            if src_lengths is not None:
                mask = sequence_mask(src_lengths)
                mask = mask.unsqueeze(1)
                scores.data.masked_fill_(1-mask, self.mask_val)
            # scoresF should be softmax
            log_scores = F.log_softmax(scores, dim=-1)
            scores = F.softmax(scores, dim=-1)

            # Make scores : T x N x S
            scores = scores.transpose(0, 1)
            log_scores = log_scores.transpose(0, 1)

            scores = Params(
                alpha=scores,
                log_alpha=log_scores,
                dist_type=self.dist_type,
            )
        elif self.dist_type == "none":
            scores = torch.bmm(tgt_memory_bank, src_memory_bank)
            # mask source attention
            if src_lengths is not None:
                mask = sequence_mask(src_lengths)
                mask = mask.unsqueeze(1)
                scores.data.masked_fill_(1-mask, self.mask_val)
            scores = Params(
                alpha= scores.transpose(0, 1),
                dist_type=self.dist_type,
            )
        else:
            raise Exception("Unsupported dist_type")

        # T x N x S
        return scores


class ViRNNDecoder(InputFeedRNNDecoder):
    def __init__(self, *args, **kwargs):
        # Hack to get get subclassing working
        p_dist_type = kwargs.pop("p_dist_type")
        q_dist_type = kwargs.pop("q_dist_type")
        use_prior = kwargs.pop("use_prior")
        scoresF = kwargs.pop("scoresF")
        n_samples = kwargs.pop("n_samples")
        mode = kwargs.pop("mode")
        super(ViRNNDecoder, self).__init__(*args, **kwargs)
        self.attn = onmt.modules.VariationalAttention(
            src_dim         = self.memory_size,
            tgt_dim         = self.hidden_size,
            attn_dim        = self.attn_size,
            p_dist_type     = p_dist_type,
            q_dist_type     = q_dist_type,
            use_prior       = use_prior,
            scoresF         = scoresF,
            n_samples       = n_samples,
            mode            = mode,
            attn_type       = kwargs["attn_type"],
        )

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None,
                          q_scores=None, tgt_emb=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        decoder_outputs_baseline = []
        dist_infos = []
        attns = {"std": []}
        if q_scores is not None:
            attns["q"] = []
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.dropout(self.embeddings(tgt)) if tgt_emb is None else tgt_emb
        assert emb.dim() == 3  # len x batch x embedding_dim

        tgt_len, batch_size =  emb.size(0), emb.size(1)
        src_len = memory_bank.size(0)

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], -1)

            rnn_output, hidden = self.rnn(decoder_input.unsqueeze(0), hidden)
            rnn_output = rnn_output.squeeze(0)
            if q_scores is not None:
                # map over tensor-like keys
                q_scores_i = Params(
                    alpha=q_scores.alpha[i],
                    log_alpha=q_scores.log_alpha[i],
                    dist_type=q_scores.dist_type,
                )
            else:
                q_scores_i = None
            decoder_output_y, decoder_output_c, context_c, attn_c, dist_info = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths,
                q_scores=q_scores_i)

            dist_infos += [dist_info]
            if self.context_gate is not None and decoder_output_c is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output_c = self.context_gate(
                    decoder_input, rnn_output, decoder_output_c
                )
            if decoder_output_c is not None:
                decoder_output_c = self.dropout(decoder_output_c)
            input_feed = context_c

            # decoder_output_y : K x N x H
            decoder_output_y = self.dropout(decoder_output_y)

            decoder_outputs += [decoder_output_y]
            if decoder_output_c is not None:
                decoder_outputs_baseline += [decoder_output_c]
            attns["std"] += [attn_c]
            if q_scores is not None:
                attns["q"] += [q_scores.alpha[i]]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]

        q_info = Params(
            alpha = q_scores.alpha,
            dist_type = q_scores.dist_type,
            samples = torch.stack([d.q.samples for d in dist_infos], dim=0)
                if dist_infos[0].q.samples is not None else None,
            log_alpha = q_scores.log_alpha,
            sample_log_probs = torch.stack([d.q.sample_log_probs for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs is not None else None,
            sample_log_probs_q = torch.stack([d.q.sample_log_probs_q for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs_q is not None else None,
            sample_log_probs_p = torch.stack([d.q.sample_log_probs_p for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_log_probs_p is not None else None,
            sample_p_div_q_log = torch.stack([d.q.sample_p_div_q_log for d in dist_infos], dim=0)
                if dist_infos[0].q.sample_p_div_q_log is not None else None,
        ) if q_scores is not None else None
        p_info = Params(
            alpha = torch.stack([d.p.alpha for d in dist_infos], dim=0),
            dist_type = dist_infos[0].p.dist_type,
            log_alpha = torch.stack([d.p.log_alpha for d in dist_infos], dim=0)
                if dist_infos[0].p.log_alpha is not None else None,
            samples = torch.stack([d.p.samples for d in dist_infos], dim=0)
                if dist_infos[0].p.samples is not None else None,
        )
        dist_info = DistInfo(
            q=q_info,
            p=p_info,
        )

        return hidden, decoder_outputs, input_feed, attns, dist_info, decoder_outputs_baseline

    def forward(self, tgt, memory_bank, state, memory_lengths=None, q_scores=None, tgt_emb=None):
        # Check
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, input_feed, attns, dist_info, decoder_outputs_baseline = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths,
            q_scores=q_scores, tgt_emb=tgt_emb)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, input_feed.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # T x K x N x H
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        if len(decoder_outputs_baseline) > 0:
            decoder_outputs_baseline = torch.stack(decoder_outputs_baseline, dim=0)
        else:
            decoder_outputs_baseline = None
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns, dist_info, decoder_outputs_baseline

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return nn.LSTM(
            num_layers=num_layers, input_size=input_size,
            hidden_size=hidden_size, dropout=dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.memory_size


class ViNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multigpu (bool): setup for multigpu support
    """
    def __init__(
        self, encoder, decoder, inference_network,
        multigpu=False, dist_type="categorical", dbg=False, use_prior=False,
        n_samples=1, k=0):
        self.multigpu = multigpu
        super(ViNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.inference_network = inference_network
        self.dist_type = dist_type
        self.dbg = dbg
        self._use_prior = use_prior
        self.n_samples = n_samples
        self.silent = False
        self.k = k

    @property
    def use_prior(self):
        return self._use_prior

    @use_prior.setter
    def use_prior(self, value):
        self._use_prior = value
        self.decoder.attn.use_prior = value

    @property
    def n_samples(self):
        return self.decoder.attn.n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value
        self.decoder.attn.n_samples = value

    @property
    def k(self):
        return self.decoder.attn.k

    @k.setter
    def k(self, value):
        self._k = value
        self.decoder.attn.k = value

    @property
    def mode(self):
        return self.decoder.attn.mode

    @mode.setter
    def mode(self, value):
        assert value in ["sample", "enum", "exact", "wsram"]
        if not self.silent:
            print("switching mode to {}".format(value))
        self.decoder.attn.mode = value

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        src_emb = self.encoder.dropout(self.encoder.embeddings(src))
        tgt_emb = self.decoder.dropout(self.decoder.embeddings(tgt))
        if self.dbg:
            # only see past
            inftgt = tgt[:-1]
        else:
            # see present
            inftgt = tgt[1:]
            inftgt_emb = tgt_emb[1:]
        tgt = tgt[:-1]  # exclude last target from inputs
        tgt_emb = tgt_emb[:-1]  # exclude last target from inputs
        tgt_length, batch_size, rnn_size = tgt.size()

        enc_final, memory_bank = self.encoder(src, lengths, emb=src_emb)
        enc_state = self.decoder.init_decoder_state(
            src, memory_bank, enc_final)
        # enc_state.* should all be 0

        if self.inference_network is not None and not self.use_prior:
            # inference network q(z|x,y)
            q_scores = self.inference_network(
                src, inftgt, lengths, src_emb=src_emb, tgt_emb=inftgt_emb) # batch_size, tgt_length, src_length
        else:
            q_scores = None
        decoder_outputs, dec_state, attns, dist_info, decoder_outputs_baseline = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths,
                         q_scores=q_scores,
                         tgt_emb=tgt_emb)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return decoder_outputs, attns, dec_state, dist_info, decoder_outputs_baseline
