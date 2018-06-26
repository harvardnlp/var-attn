# Latent Alignment and Variational Attention

This is a [Pytorch](https://github.com/pytorch/pytorch)
implementation of the paper [Latent Alignment and Variational Attention](https://arxiv.org/abs/1802.02550)
from a fork of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).


## Dependencies

The code was tested with `python 3.6` and `pytorch 0.4`.
To install the dependencies, run
```bash
pip install -r requirements.txt
```

## Running the code
All commands are in the script `va.sh`.

To preprocess the data, run
```bash
source va.sh && preprocess_bpe
```
The raw data in `data/iwslt14-de-en` was obtained from the
[fairseq](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh) repo
with `BPE_TOKENS=14000`.

### Training the model
To train a model, run one of the following commands:
* Variational categorical attention with REINFORCE
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_sample_b6
```
* Variational categorical attention with exact ELBO
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_enum_b6
```
* Categorical attention with exact evidence
```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 train_exact_b6
```

### Evaluating on test
The exact perplexity of the generative model can be obtained by running
the following command with `$model` replaced with a saved checkpoint.

```bash
source va.sh && CUDA_VISIBLE_DEVICES=0 eval_cat $model
```

