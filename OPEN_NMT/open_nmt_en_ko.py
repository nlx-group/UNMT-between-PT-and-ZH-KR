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



yaml_config =  """

save_data: en-ko/data_en_ko/vocab
src_vocab: en-ko/data_en_ko/vocab.src
tgt_vocab: en-ko/data_en_ko/vocab.tgt
src_vocab_size: 32000
tgt_vocab_size: 32000

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: en-ko/data_en_ko/source.model
src_subword_nbest:
tgt_subword_type: sentencepiece
tgt_subword_model: en-ko/data_en_ko//target.model
tgt_subword_nbest:

# Number of candidates for SentencePiece sampling
subword_nbest: 20
# Smoothing parameter for SentencePiece sampling
subword_alpha: 0.1
# Specific arguments for pyonmttok
src_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"
tgt_onmttok_kwargs: "{'mode': 'none', 'spacer_annotate': True}"

#log file
log_file: en-ko/train.log
#train on a single gpu
world_size: 1
gpu_ranks: [0]
skip_empty_level: silent

#save checkpoints
save_model: en-ko/save_en_ko/model
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

#corpus_opt

data:
    Paracrawl:
        path_src: en-ko/Paracrawl/paracrawl_en.txt-filtered.en.train
        path_tgt: en-ko/Paracrawl/paracrawl_ko.txt-filtered.ko.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-ko/Paracrawl/paracrawl_en.txt-filtered.en.dev
        path_tgt: en-ko/Paracrawl/paracrawl_ko.txt-filtered.ko.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    Tanzil:
        path_src: en-ko/Tanzil/Tanzil.en.txt-filtered.en.train
        path_tgt: en-ko/Tanzil/Tanzil.ko.txt-filtered.ko.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-ko/Tanzil/Tanzil.en.txt-filtered.en.dev
        path_tgt: en-ko/Tanzil/Tanzil.ko.txt-filtered.ko.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    ted:
        path_src: en-ko/ted/TED2020.en-ko.en-filtered.en.train
        path_tgt: en-ko/ted/TED2020.en-ko.ko-filtered.ko.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-ko/ted/TED2020.en-ko.en-filtered.en.subword.dev
        path_tgt: en-ko/ted/TED2020.en-ko.ko-filtered.ko.subword.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    wiki_matriz:
        path_src: en-ko/wikimatrix/WikiMatrix.en.txt-filtered.en.train
        path_tgt: en-ko/wikimatrix/WikiMatrix.ko.txt-filtered.ko.train
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-ko/wikimatrix/WikiMatrix.en.txt-filtered.en.dev
        path_tgt: en-ko/wikimatrix/WikiMatrix.ko.txt-filtered.ko.dev
        src_seq_length: 150
        tgt_seq_length: 150
        transforms: [sentencepiece, filtertoolong]
        weight: 1
       
"""

config = yaml.safe_load(yaml_config) #ver o que isto faz
# write yaml_config to a file
with open("en-ko/data_en_ko/config.yaml", 'w') as f:
    f.write(yaml_config)

#STEP1: BUILD VOCAB
bash(" onmt_build_vocab " +
    " -config " + quote("en-ko/data_en_ko/config.yaml")
    + ' -n_sample 1000  ')

# STEP2: TRAIN
bash(' onmt_train ' + 
    ' -config ' + quote("en-ko/data_en_ko/config.yaml")
    + ' --train_from en-ko/save_en_ko/model_step_330000.pt')

# # STEP3: TRANSLATE
bash (' onmt_translate ' +
' -model ' + quote('en-ko/save_en_ko/model')+
' -src ' + quote('en-ko/CCAligned/CCAligned.en.txt-filtered.en.subword.test') + 
' -output ' + quote('en-ko/save_en_ko/pred_1000.txt') +
' -gpu 0' +
' -verbose')

# STEP4: Evaluation 

bash(' python3 ' + ' compute-bleu.py ' + '  ' + ' ')

