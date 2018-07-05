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
    model=$1
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

# Transformer
# Shadow variables
TEXT=data/wmt14-de-en
DATA=data/wmt/wmt_125
DATATEST=data/wmt/wmt_125_test

preprocess_bpe_wmt(){
    # Preprocesses the data in data/iwslt14-de-en
    # Since we are using BPE, we do not force any unks.
    mkdir -p data/wmt
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

train_soft_trans() {
    python  train.py -data /tmp/de2/data -save_model /tmp/extra -gpuid 1 \
        -layers 6 -rnn_size 512 -word_vec_size 512   \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 100000  -max_generator_batches 32 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1  
}
