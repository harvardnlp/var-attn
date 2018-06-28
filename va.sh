TEXT=data/iwslt14-de-en
DATA=data/iwslt/iwslt_125
DATATEST=data/iwslt/iwslt_125_test

preprocess_bpe(){
    # Preprocesses the data in data/iwslt14-de-en
    # Since we are using BPE, we do not force any unks.
    mkdir data/iwslt
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
        -inference_network_rnn_size 1024 \
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
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_cat_enum_b6() {
    gpuid=0
    seed=131
    name=model_cat_enum_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode enum \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 1024 \
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
        -n_samples 1 \
        -start_decay_at 2 \
        -learning_rate_decay 0.5 \
        -report_every 1000 | tee $name.log
}

train_exact_b6() {
    gpuid=0
    seed=131
    name=model_exact_b6
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -mode exact \
        -use_generative_model 1 \
        -batch_size 6 \
        -encoder_type brnn \
        -inference_network_type bigbrnn \
        -inference_network_rnn_size 1024 \
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
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

eval_cat() {
    model=$1
    model=model_soft_b8_dbg_acc_63.16_ppl_7.29_e8.pt
    python train.py \
        -data /n/rush_lab/users/yuntian/latent_attention/normal/data/iwslt_125_test \
        -eval_with $model \
        -save_model none -gpuid 0 -seed 131 -encoder_type brnn -batch_size 8 \
        -accum_count 1 -valid_batch_size 2 -epochs 30 -inference_network_type brnn \
        -p_dist_type categorical -q_dist_type categorical -alpha_transformation sm \
        -global_attention mlp \
        -optim adam -learning_rate 3e-4 -n_samples 1 -mode sample \
        -eval_only 1
}

gen_cat() {
    # VAE Enum
    model=/n/rush_lab/jc/onmt-attn/iwslt14-de-en/models/model_cat_enum_b8_dbg/model_cat_enum_b8_dbg_acc_74.47_ppl_3.82_e7.pt
    # VAE Sample
    model=/n/rush_lab/jc/onmt-attn/iwslt14-de-en/models/model_cat_sample_b8_dbg/model_cat_sample_b8_dbg_acc_73.44_ppl_3.94_e15.pt
    # Soft
    model=model_soft_b8_dbg_acc_63.16_ppl_7.29_e8.pt
        #-alpha 1 \
    python -m pdb translate.py \
        -src data/iwslt14-de-en/test.de.bpe \
        -beam_size 10 \
        -batch_size 2 \
        -gpu 0 \
        -output $model.out \
        -model $model
}
# sed "s/@@ //g" /n/rush_lab/jc/onmt-attn/iwslt14-de-en/models/model_cat_enum_b8_dbg/model_cat_enum_b8_dbg_acc_74.47_ppl_3.82_e7.pt.out | perl tools/multi-bleu.perl data/iwslt14-de-en/test.en

### DEBUG

soft_dbg() {
    # The parameters for the soft model are slightly different
    seed=131
    name=model_soft_b6
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
        -learning_rate 3e-4 \
        -start_decay_at 2 \
        -global_attention mlp \
        -dropout 0 \
        -report_every 1000
}

train_soft_b8_dbg() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b8_dbg
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 8 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -adam_eps 1e-6  \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

train_soft_b8_dbg_adam() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b8_dbg_adam
    gpuid=0
    python train.py \
        -data $DATA \
        -save_model $name -gpuid $gpuid -seed $seed \
        -src_word_vec_size 512 \
        -tgt_word_vec_size 512 \
        -memory_size 1024 \
        -decoder_rnn_size 768 \
        -attention_size 512 \
        -encoder_type brnn -batch_size 8 \
        -accum_count 1 -valid_batch_size 32 \
        -epochs 30 -optim adam \
        -learning_rate 3e-4 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 1000 | tee $name.log
}

yoon_soft() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        python train_attn_var2.py \
        --gpu 0 \
        --checkpoint_path yoon-chp.pt \
        --attn soft \
        --print_every 1000 | tee yoon_soft_b8.log
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

train_soft_b32_dbg_adam() {
    # The parameters for the soft model are slightly different
    seed=3435
    name=model_soft_b32_dbg_adam
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
        -adam_eps 1e-6 \
        -start_decay_at 2 \
        -global_attention mlp \
        -report_every 500 | tee $name.log
}

yoon_soft_b32() {
    PYTHONPATH=/n/rush_lab/users/yoonkim/seq2seq-py \
        python train_attn_var2.py \
        --train_file /n/rush_lab/users/yoonkim/seq2seq-py/data/bpe/iwslt-bpe-batch32-train.hdf5 \
        --gpu 0 \
        --checkpoint_path yoon-chp-b32.pt \
        --print_every 500 \
        --attn soft | tee yoon_soft_b32.log
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
        -inference_network_rnn_size 8 \
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
        -report_every 1000
}
