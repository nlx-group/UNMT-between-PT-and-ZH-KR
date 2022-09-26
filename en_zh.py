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


"""
The on the fly stuff is only for training for now. 
For inference, you need to tokenize your source text prior to calling onmt_translate.
After you segment your source and target files with the generated SentencePiece models,
you must build vocab using OpenNMT-py to generate vocab files compatible with it. 
Basically you do subword and then use that model to train your data.
((What i could do. Provide a tokenized dataset with jieba))
"""
yaml_config =  """
save_data: en-zh/data/vocab
src_vocab: en-zh/data/vocab.src
tgt_vocab: en-zh/data/vocab.tgt
src_vocab_size: 32000
tgt_vocab_size: 32000

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: en-zh/bpe/en.model # train on that script
src_subword_nbest:
tgt_subword_type: sentencepiece
tgt_subword_model: en-zh/bpe/zh.model #train on that script
tgt_subword_nbest:

# Number of candidates for SentencePiece sampling
subword_nbest: 20
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'aggressive', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'None', 'spacer_annotate': True}"
#filter
src_seq_length: 120
tgt_seq_length: 120



skip_empty_level: silent


#train on a single gpu
world_size: 1
gpu_ranks: [0]

#log file
log_file: train.log

#save checkpoints
save_model: en-zh/save/model
save_checkpoint_steps: 500
save_checkpoint_steps: 10000
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
#4096
batch_size: 500
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

#corpus_opt

data:
    Tanzil:
        path_src: en-zh/Tanzil/Tanzil.en-zh.en
        path_tgt: en-zh/Tanzil/Tanzil.en-zh.zh.tok
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    valid:
        path_src: en-zh/Tanzil/Tanzil.en-zh.en.dev
        path_tgt: en-zh/Tanzil/Tanzil.en-zh.zh.tok.dev
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    NewsCommentary:
        path_src: en-zh/NewsCommentary/News-Commentary.en-zh.en.train
        path_tgt: en-zh/NewsCommentary/News-Commentary.en-zh.zh.tok.train
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    valid:
        path_src: en-zh/NewsCommentary/News-Commentary.en-zh.en.dev
        path_tgt: en-zh/NewsCommentary/News-Commentary.en-zh.zh.tok.dev
        weight: 1
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    ted:
        path_src: en-zh/Tanzil/Tanzil.en-zh.en.train
        path_tgt: en-zh/Tanzil/Tanzil.en-zh.zh.tok.train
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    valid: 
        path_src: en-zh/Tanzil/Tanzil.en-zh.en.dev
        path_tgt: en-zh/Tanzil/Tanzil.en-zh.zh.tok.dev
        transformers: [onmt_tokenize, sentencepiece,filtertoolong]
        weight: 1
    
"""

config = yaml.safe_load(yaml_config) #ver o que isto faz
# write yaml_config to a file
with open("en-zh/save/config.yaml", 'w') as f:
    f.write(yaml_config)

#STEP1: BUILD VOCAB
bash(" onmt_build_vocab " +
    " -config " + quote("en-zh/save/config.yaml")
    + ' -n_sample 10000 ')

# STEP2: TRAIN
bash(' onmt_train ' + 
    ' -config ' + quote("en-zh/save/config.yaml")
    + ' --train_from ' + quote('en-zh/save/model_step_120000.pt'))



# STEP3: TRANSLATE
bash (' onmt_translate ' +
' -model ' + quote('en-zh/save/model_step_1000.pt')+
' -src ' + quote('en-zh/NewsCommentary/News-Commentary.en-zh.en.test') + 
' -output ' + quote('pred_1000.txt') +
' -gpu 0' +
' -verbose'
)

#STEP4: INFERENCE
bash(' onmt_release_model ' + 
'-model ' + quote(' en-zh/save/model_step_1000.pt ') + 
' -output ' + quote(' en-zh/save/model_step_1000_release.pt '))
