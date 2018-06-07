import argparse

def get_args():
    args = argparse.ArgumentParser()
    #args.add_argument("--output", type=str)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--bsz", type=int, default=16)
    args.add_argument("--start", type=float, default=0)
    args.add_argument("--steps", type=int, default=0)
    args.add_argument("--q-norm", type=str, default="bn")
    args.add_argument("--p-norm", type=str, default="none")
    return args.parse_args()

args = get_args()

lr = args.lr
bsz = args.bsz
start = args.start
steps = args.steps
qn = args.q_norm
pn = args.p_norm

#output = "train_iwslt14deen_demi_dir_gen_brnn_10samples_detachpkl"
#output = "train_iwslt14deen_demi_dir_gen_brnn_5samples_ignorekl",
#output = "train_iwslt14deen_demi_dir_gen_brnn_1sample_detachpkl",
#output = "train_iwslt14deen_demi_bn_exp_dir_brnn_adam_1sample"
#output = "train_iwslt14deen_demi_bn_sp_dir_brnn_adam_1sample"

output = "train_bpe_iwslt14deen_demi_ln_brnn_adam_1sample"

header_kv = dict(
    #queue = "seas_dgx1_priority",
    queue = "seas_dgx1",
    ngpus = 1,
    #output = "{}.lr{}.bsz{}.alpha{}-{}.qn{}.pn{}".format(output, lr, bsz, start, steps, qn, pn),
    output = "lol",
)
header = """#!/bin/bash
#SBATCH -p {queue}
#SBATCH --gres=gpu:{ngpus}
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 64000
#SBATCH -t 0-23:59 # D-HH:MM
#SBATCH -o {output}
#SBATCH -J {output}
""".format(**header_kv)

prelude = """
source /n/home13/jchiu/.bash_profile
pym9env
"""

body = """
python /n/home13/jchiu/projects/OpenNMT-py/train.py \
    -data /n/rush_lab/users/yuntian/latent_attention/normal/data/iwslt_125 -save_model model \
    -gpuid 0 -seed 131 -encoder_type brnn -batch_size 16 -accum_count 1 -valid_batch_size 64 \
    -epochs 29 -inference_network_type brnn \
    -p_dist_type categorical -q_dist_type categorical -alpha_transformation sm \
    -optim adam -learning_rate 3e-4
""".format(output, lr, bsz, start, steps, qn, pn)

print(header + prelude + body)
