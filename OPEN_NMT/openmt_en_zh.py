#imports
import subprocess
import argparse
import os
from shlex import quote
import yaml
from argparse import Namespace
from collections import defaultdict, Counter


ROOT = os.path.dirname(os.path.abspath(__file__))

def bash(command):
    subprocess.run(['bash', '-c', command])



#Prepare data and vocab
yaml_config =  """
save_data: en-zh/data_en_zh/vocab
src_vocab: en-zh/data_en_zh/vocab.src
tgt_vocab: en-zh/data_en_zh/vocab.tgt
src_vocab_size: 32000
tgt_vocab_size: 32000

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: en-zh/data_en_zh/source.model
src_subword_nbest:
tgt_subword_type: sentencepiece
tgt_subword_model: en-zh/data_en_zh/target.model
tgt_subword_nbest:

# Number of candidates for SentencePiece sampling
subword_nbest: 20
# Smoothing parameter for SentencePiece sampling
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"

#train on a single gpu
world_size: 1
gpu_ranks: [0]

#log file
log_file: en-zh/train.log

#save checkpoints
save_model: en-zh/save/model
save_checkpoint_steps: 500
save_checkpoint_steps: 1000
keep_checkpoint: 10
seed: 3435
train_steps: 500000
valid_steps: 10000
warmup_steps: 8000
report_every: 100

decoder_type: transformer
encoder_type: transformer
word_vec_size: 512
rnn_size: 512
layers: 6
transformer_ff: 2048
heads: 8

accum_count: 8
optim: adam
adam_beta1: 0.9
adam_beta2: 0.998
decay_method: noam
learning_rate: 2.0
max_grad_norm: 0.0

batch_size: 4096
batch_type: tokens
normalization: tokens
dropout: 0.1
label_smoothing: 0.1

max_generator_batches: 2

param_init: 0.0
param_init_glorot: 'true'
position_encoding: 'true'

world_size: 1
gpu_ranks: [0]
# silently ignore empty lines in the data
skip_empty_level: silent
#corpus_opt

data:
    wikimatriz:
        path_src: en-zh/wikimatrix/WikiMatrix.en-zh.en.train
        path_tgt: en-zh/wikimatrix/WikiMatrix.en-zh.zh.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-zh/wikimatrix/WikiMatrix.en-zh.en.dev
        path_tgt: en-zh/wikimatrix/WikiMatrix.en-zh.zh.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    tanzil:
        path_src: en-zh/tanzil/Tanzil.en-zh.en.train
        path_tgt: en-zh/tanzil/Tanzil.en-zh.zh.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-zh/tanzil//Tanzil.en-zh.en.dev
        path_tgt: en-zh/tanzil/Tanzil.en-zh.zh.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    wmt:
        path_src: en-zh/wmt/WMT-News.en-zh.en.train
        path_tgt: en-zh/wmt/WMT-News.en-zh.zh.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1 
    valid:
        path_src: en-zh/wmt/WMT-News.en-zh.en.dev
        path_tgt: en-zh/wmt/WMT-News.en-zh.zh.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1 
    newscommentary:
        path_src: en-zh/NewsCommentary/News-Commentary.en-zh.en.train
        path_tgt: en-zh/NewsCommentary/News-Commentary.en-zh.zh.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1 
    valid:
        path_src: en-zh/NewsCommentary/News-Commentary.en-zh.en.dev
        path_tgt: en-zh/NewsCommentary/News-Commentary.en-zh.zh.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1 


"""

config = yaml.safe_load(yaml_config) #ver o que isto faz
# write yaml_config to a file
with open("config.yaml", 'w') as f:
    f.write(yaml_config)

#STEP1: BUILD VOCAB
bash(" onmt_build_vocab " +
    " -config " + quote("config.yaml")
    + ' -n_sample 1000 ')

# STEP2: TRAIN
bash(' onmt_train ' + 
    ' -config ' + quote("config.yaml") + 
    ' --train_from en-zh/save/model_step_229000.pt')

# # STEP3: TRANSLATE
bash (' onmt_translate ' +
' -model ' + quote('en-zh/save/model_step.zh_step_500000.pt')+
' -src ' + quote('en-zh/ted/TED2020.en-zh_cn.en.subword.test') + 
' -output ' + quote('pred_en_zh.txt') +
' -verbose')


#STEP4 : EVALUATION

bash(' python3 ' + ' compute-bleu.py ' + 'en-zh/ted/TED2020.en-zh_cn.en.subword.test' + ' pred_en_zh.txt')