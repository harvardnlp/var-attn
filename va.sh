TEXT=data/iwslt14-de-en
DATA=data/iwslt/iwslt_125
DATATEST=data/iwslt/iwslt_125_test

preprocess_bpe(){
    # Preprocesses the data in data/iwslt14-de-en
    # Since we are using BPE, we do not force any unks.
    mkdir -p data/iwslt
    python preprocess.py \
        -train_src ${TEXT}/train.de.bpe -train_tgt ${TEXT}/train.en.bpe \
        -valid_src ${TEXT}/valid.de.bpe -valid_tgt ${TEXT}/valid.en.bpe \
        -src_vocab_size 80000 -tgt_vocab_size 80000 \
        -src_words_min_frequency 0 -tgt_words_min_frequency 0 \
        -src_seq_length 125 -tgt_seq_length 125 \
        -save_data $DATA

    # Get the test data for evaluation
    python preprocess.py \
        -train_src ${TEXT}/train.de.bpe -train_tgt ${TEXT}/train.en.bpe \
        -valid_src ${TEXT}/test.de.bpe -valid_tgt ${TEXT}/test.en.bpe \
        -src_vocab_size 80000 -tgt_vocab_size 80000 \
        -src_words_min_frequency 0 -tgt_words_min_frequency 0 \
        -src_seq_length 125 -tgt_seq_length 125 \
        -leave_valid \
        -save_data $DATATEST
}

train_cat_sample_b6() {
    gpuid=0
    seed=3435
    name=model_cat_sample_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode sample \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_cat_sample_b32() {
    gpuid=0
    seed=3435
    name=model_cat_sample_b32
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode sample \
        -batch_size 32 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 500 | tee $name.log
}

train_cat_enum_b6() {
    gpuid=0
    seed=3435
    name=model_cat_enum_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode enum \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_exact_b6() {
    gpuid=0
    seed=3435
    name=model_exact_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode exact \
        -use_generative_model 1 \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 512 \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -accum_count 1 \
        -valid_batch_size 2 \
        -epochs 30 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_soft_b6() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b6
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

eval_cat() {
    model=$1
    # Soft
    model=model_soft_b6_dbg_dropout_acc_64.89_ppl_6.59_e11.pt
    # Exact
    #model=model_exact_b6_acc_65.18_ppl_5.82_e11.pt
    # VAE Enum
    #model=model_cat_enum_b6_acc_75.20_ppl_6.23_e10.pt
    # VAE Sample
    #model=model_cat_sample_b6_acc_74.52_ppl_6.53_e12.pt
    python train.py \
        -data $DATATEST \
        -eval_with $model \
        -save_model none -gpuid 0 -seed 131 -encoder_type brnn -batch_size 8 \
        -accum_count 1 -valid_batch_size 2 -epochs 30 -inference_network_type bigbrnn \
        -p_dist_type categorical -q_dist_type categorical -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 -n_samples 1 -mode sample \
        -eval_only 1
}

gen_cat() {
    # Soft
    model=model_soft_b6_dbg_dropout_acc_64.89_ppl_6.59_e11.pt
    # Exact
    model=model_exact_b6_acc_65.18_ppl_5.82_e11.pt
    # VAE Enum
    model=model_cat_enum_b6_acc_75.20_ppl_6.23_e10.pt
    # VAE Sample
    model=model_cat_sample_b6_acc_74.52_ppl_6.53_e12.pt
    python translate.py \
        -src data/iwslt14-de-en/test.de.bpe \
        -beam_size 10 \
        -batch_size 2 \
        -length_penalty wu \
        -alpha 1 \
        -eos_norm 3 \
        -gpu 0 \
        -output $model.out \
        -model $model
}
# sed "s/@@ //g" /n/rush_lab/jc/onmt-attn/iwslt14-de-en/models/model_cat_enum_b6_dbg/model_cat_enum_b6_dbg_acc_74.47_ppl_3.82_e7.pt.out | perl tools/multi-bleu.perl data/iwslt14-de-en/test.en

### DEBUG

soft_dbg() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=lol
    gpuid=0
    DBG=1 python -m pdb train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 4 \
        -tgt_word_vec_size 4 \
        -memory_size 8 \
        -decoder_rnn_size 4 \
        -attention_size 4 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -adam_eps 1e-8 \
        -learning_rate 3e-4 \
        -start_decay_at 2 \
        -global_attention mlp \
        -dropout 0 \
        -report_every 1000
}

train_soft_b6_dbg() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b6_dbg
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8  \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

train_soft_b32_dbg() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b32_dbg
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 32 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 500 | tee $name.log
}

yoon_soft() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --gpu 0 \
        --checkpoint_path yoon-chp.pt \
        --attn soft \
        --print_every 1000 | tee yoon_soft_b6.log
}

yoon_soft_b32() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --train_file /n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-batch32-train.hdf5 \
        --gpu 0 \
        --checkpoint_path yoon-chp-b32.pt \
        --print_every 500 \
        --attn soft | tee yoon_soft_b32.log
}

train_soft_b32_dbg_dropout() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b32_dbg_dropout
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 32 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 500 | tee $name.log
}

train_soft_b6_dbg_dropout() {
    seed=3435
    name=model_soft_b6_dbg_dropout
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-8 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

vae_dbg() {
    # The parameters for the soft model are slightly different
    seed=131
    name=model_cat_enum_b6
    gpuid=0
    DBG=1 python -m pdb train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -encoder_type brnn \
        -inference_network_rnn_size 4 \
        -p_dist_type categorical \
        -q_dist_type categorical \
        -alpha_transformation sm \
        -src_word_vec_size 4 \
        -tgt_word_vec_size 4 \
        -mode enum \
        -inference_network_type bigbrnn \
        -memory_size 8 \
        -decoder_rnn_size 4 \
        -attention_size 4 \
        -encoder_type brnn -batch_size 6 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -start_decay_at 2 \
        -global_attention mlp \
        -dropout 0 \
        -inference_network_dropout 0 \
        -report_every 1000
}

yoon_exact() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --gpu 0 \
        --checkpoint_path yoon-chp-exact.pt \
        --attn hard \
        --print_every 1000 | tee yoon_exact_b6.log
}

yoon_vae() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --gpu 0 \
        --checkpoint_path yoon-chp-vae.pt \
        --attn vae \
        --print_every 1000 | tee yoon_vae_b6.log
}

yoon_vaesample() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --gpu 0 \
        --checkpoint_path yoon-chp-vae-sample.pt \
        --attn vae_sample \
        --print_every 1000 | tee yoon_vae_sample_b6.log
}

yoon_sample_b32() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        stdbuf -o0 \
        python train_attn_var2.py \
        --train_file /n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-batch32-train.hdf5 \
        --gpu 0 \
        --checkpoint_path yoon-chp-vae-sample-b32.pt \
        --print_every 500 \
        --attn vae_sample | tee yoon_vae_sample_b32.log
}

eval_yoon() {
    model=yoon-chp-b32.pt
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        python train_attn_var2.py \
        --train_file /n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-test-train.hdf5 \
        --val_file /n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-test-val.hdf5 \
        --train_from $model \
        --mode test
}
