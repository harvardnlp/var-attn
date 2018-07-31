"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, \
                        Generator
from onmt.ViModels import InferenceNetwork, ViRNNDecoder, ViNMTModel
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu
from torch.nn.init import xavier_uniform

scoresF_dict = {
    "softplus": F.softplus,
    "exp": lambda x: x.clamp(-10, 10).exp(),
    #"exp": lambda x: x.exp(),
    "relu": lambda x: x.clamp(min=1e-2),
    "sm": lambda x: F.softmax(x, dim=-1),
}

def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True,
                    for_inference_network=False):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        if not for_inference_network:
            embedding_dim = opt.src_word_vec_size
        else:
            embedding_dim = opt.inference_network_src_word_vec_size
    else:
        if not for_inference_network:
            embedding_dim = opt.tgt_word_vec_size
        else:
            embedding_dim = opt.inference_network_tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]
    if not for_inference_network:
        dropout = opt.dropout
    else:
        dropout = opt.inference_network_dropout

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.memory_size, opt.decoder_rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def make_inference_network(opt, src_embeddings, tgt_embeddings,
                           src_dict, src_feature_dicts,
                           tgt_dict, tgt_feature_dicts):
    print ('Making inference network:')
    if not opt.inference_network_share_embeddings:
        print ('    * share embeddings: False')
        src_embeddings = make_embeddings(opt, src_dict,
                                         src_feature_dicts,
                                         for_inference_network=True)
        tgt_embeddings = make_embeddings(opt, tgt_dict,
                                         tgt_feature_dicts, for_encoder=False,
                                         for_inference_network=True)
    else:
        print ('    * share embeddings: True')

    inference_network_type = opt.inference_network_type
    inference_network_src_layers = opt.inference_network_src_layers
    inference_network_tgt_layers = opt.inference_network_tgt_layers
    rnn_type = opt.rnn_type
    rnn_size = opt.inference_network_rnn_size
    dropout = opt.inference_network_dropout
    scoresFstring = opt.alpha_transformation
    scoresF = scoresF_dict[scoresFstring]
    attn_type = opt.q_attn_type

    print ('    * inference network type: %s'%inference_network_type)
    print ('    * inference network RNN type: %s'%rnn_type)
    print ('    * inference network RNN size: %s'%rnn_size)
    print ('    * inference network dropout: %s'%dropout)
    print ('    * inference network src layers: %s'%inference_network_src_layers)
    print ('    * inference network tgt layers: %s'%inference_network_tgt_layers)
    print ('    * inference network alpha trans: %s'%scoresFstring)
    print ('    * inference network attn type: %s'%attn_type)
    print ('    * inference network dist type: %s'%opt.q_dist_type)
    print ('    * TODO: RNN\'s could be possibly shared')

    return InferenceNetwork(inference_network_type,
                            src_embeddings, tgt_embeddings,
                            rnn_type, inference_network_src_layers,
                            inference_network_tgt_layers, rnn_size, dropout,
                            attn_type=opt.q_attn_type,
                            dist_type=opt.q_dist_type,
                            scoresF=scoresF)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed and opt.inference_network_type == "none":
        print("input feed")
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers,
                                   opt.memory_size,
                                   opt.decoder_rnn_size,
                                   opt.attention_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    elif opt.input_feed and opt.inference_network_type != "none":
        print("VARIATIONAL DECODER")
        scoresFstring = opt.alpha_transformation
        scoresF = scoresF_dict[scoresFstring]

        return ViRNNDecoder(
            opt.rnn_type, opt.brnn,
            opt.dec_layers,
            memory_size     = opt.memory_size,
            hidden_size     = opt.decoder_rnn_size,
            attn_size       = opt.attention_size,
            attn_type       = opt.global_attention,
            coverage_attn   = opt.coverage_attn,
            context_gate    = opt.context_gate,
            copy_attn       = opt.copy_attn,
            dropout         = opt.dropout,
            embeddings      = embeddings,
            reuse_copy_attn = opt.reuse_copy_attn,
            p_dist_type     = opt.p_dist_type,
            q_dist_type     = opt.q_dist_type,
            use_prior       = opt.use_generative_model > 0,
            scoresF         = scoresF,
            n_samples       = opt.n_samples,
            mode            = opt.mode,
        )
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        src_feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         src_feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    tgt_feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     tgt_feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make inference network.
    inference_network = make_inference_network(
        model_opt,
        src_embeddings, tgt_embeddings,
        src_dict, src_feature_dicts,
        tgt_dict, tgt_feature_dicts
    ) if model_opt.inference_network_type != "none" else None

    # Make NMTModel(= encoder + decoder + inference network).
    model = (
        NMTModel(encoder, decoder)
        if inference_network is None
        else ViNMTModel(
            encoder, decoder,
            inference_network,
            n_samples=model_opt.n_samples,
            dist_type=model_opt.p_dist_type,
            dbg=model_opt.dbg_inf,
            use_prior=model_opt.use_generative_model > 0)
    )
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        """
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax(dim=1))
        """
        generator = Generator(
            in_dim = model_opt.decoder_rnn_size,
            out_dim = len(fields["tgt"].vocab),
            mode = model_opt.mode,
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        #model.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu >= 0:
        model.cuda()
    else:
        model.cpu()

    return model
