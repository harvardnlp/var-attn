# Latent Alignment and Variational Attention

This is a [Pytorch](https://github.com/pytorch/pytorch)
implementation of the paper [Latent Alignment and Variational Attention](https://arxiv.org/abs/1807.03756)
from a fork of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).


## Dependencies

The code was tested with `python 3.6` and `pytorch 0.4`.
To install the dependencies, run
```bash
pip install -r requirements.txt
```

## Running the code
All commands are in the script `va.sh`.

### Preprocessing the data
To preprocess the data, run
```bash
source va.sh && preprocess_bpe
```
The raw data in `data/iwslt14-de-en` was obtained from the
[fairseq](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh) repo
with `BPE_TOKENS=14000`.

### Training the model
To train a model, run one of the following commands:
* Soft attention
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_soft_b6
```
* Categorical attention with exact evidence
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_exact_b6
```
* Variational categorical attention with exact ELBO
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_enum_b6
```
* Variational categorical attention with REINFORCE
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_sample_b6
```
* Variational categorical attention with Gumbel-Softmax
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_gumbel_b6
```
* Variational categorical attention using Wake-Sleep algorithm (Ba et al 2015)
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_wsram_b6
```
Checkpoints will be saved to the project's root directory.

### Evaluating on test
The exact perplexity of the generative model can be obtained by running
the following command with `$model` replaced with a saved checkpoint.
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 eval_cat $model
```

The model can also be used to generate translations of the test data:
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 gen_cat $model
sed -e "s/@@ //g" $model.out | perl tools/multi-bleu.perl data/iwslt14-de-en/test.en
```

## Trained Models
Models with the lowest validation PPL were selected for evaluation on test.
Numbers are slightly different from those reported in the paper since this is a re-implementation.

| Model | Test PPL  | Test BLEU |
| ----- | --------: | --------: |
| [Soft Attention](http://lstm.seas.harvard.edu/latex/var_attn/model_soft_b6_acc_64.89_ppl_6.59_e11.pt) | 7.17  | [32.77](http://lstm.seas.harvard.edu/latex/var_attn/model_soft_b6_acc_64.89_ppl_6.59_e11.pt.out) |
| [Exact Marginalization](http://lstm.seas.harvard.edu/latex/var_attn/model_exact_b6_acc_65.18_ppl_5.82_e11.pt) | 6.34 | [33.29](http://lstm.seas.harvard.edu/latex/var_attn/model_exact_b6_acc_65.18_ppl_5.82_e11.pt.out) |
| [Variational Attention + Enumeration](http://lstm.seas.harvard.edu/latex/var_attn/model_cat_enum_b6_acc_75.20_ppl_6.23_e10.pt) | 6.08  | [33.69](http://lstm.seas.harvard.edu/latex/var_attn/model_cat_enum_b6_acc_75.20_ppl_6.23_e10.pt.out) |
| [Variational Attention + Sampling](http://lstm.seas.harvard.edu/latex/var_attn/model_cat_sample_b6_acc_74.52_ppl_6.53_e12.pt) | 6.17 | [33.30](http://lstm.seas.harvard.edu/latex/var_attn/model_cat_sample_b6_acc_74.52_ppl_6.53_e12.pt.out) |
