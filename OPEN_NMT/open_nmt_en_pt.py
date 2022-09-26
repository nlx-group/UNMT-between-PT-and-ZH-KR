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
save_data: data/vocab
src_vocab: data/vocab.src
tgt_vocab: data/vocab.tgt
src_vocab_size: 32000
tgt_vocab_size: 32000

# Tokenization options
src_subword_type: sentencepiece
src_subword_model: pt.model
src_subword_nbest:
tgt_subword_type: sentencepiece
tgt_subword_model: en_pt.model
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
log_file: train.log

#save checkpoints
save_model: save/model
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
# silently ignore empty lines in the data
skip_empty_level: silent
#corpus_opt

data:
    scielo:
        path_src: en-pt/scielo/scielo_pt.txt
        path_tgt: en-pt/scielo/scielo_en
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 3
    valid:
        path_src: en-pt/scielo/scielo_val.pt.txt
        path_tgt: en-pt/scielo/scielo_val_en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
    europarl:
        path_src: en-pt/europarl/europarl_pt
        path_tgt: en-pt/europarl/europarl_en
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 2
    valid:
        path_src: en-pt/europarl/europarl_val.pt.txt
        path_tgt: en-pt/europarl/europarl_val.en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
    wikipedia:
        path_src: en-pt/wikipedia/wikipedia_pt
        path_tgt: en-pt/wikipedia/wikipedia_en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 2
    valid:
        path_src: en-pt/wikipedia/Wikipedia.pt_val.txt
        path_tgt: en-pt/wikipedia/Wikipedia.en_val.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
    ted:
        path_src: en-pt/TED/ted_pt.txt
        path_tgt: en-pt/TED/ted_en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 30
    valid:
        path_src: en-pt/TED/ted_val.pt.txt
        path_tgt: en-pt/TED/ted_val.en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
    wiki_matriz:
        path_src: en-pt/wikimatrix/wikimatrix_pr.txt
        path_tgt: en-pt/wikimatrix/wikimatrix_en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 2
    valid:
        path_src: en-pt/wikimatrix/wikimatrix_val.pt.txt
        path_tgt: en-pt/wikimatrix/wikimatrix_val.en.txt
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
    paracrawl:
        path_src: en-pt/paracrawl/ParaCrawl.en-pt.pt
        path_tgt: en-pt/paracrawl/ParaCrawl.en-pt.en
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
        weight: 1
    valid:
        path_src: en-pt/paracrawl/ParaCrawl_pt_val
        path_tgt: en-pt/paracrawl/ParaCrawl_en_val
        src_seq_length: 10000
        tgt_seq_length: 10000
        src_seq_length_trunc: 400
        tgt_seq_length_trunc: 100
        transforms: [sentencepiece, filtertoolong]
                        
    

    
"""

config = yaml.safe_load(yaml_config) #ver o que isto faz
# write yaml_config to a file
with open("config.yaml", 'w') as f:
    f.write(yaml_config)

#STEP1: BUILD VOCAB
bash(" onmt_build_vocab " +
    " -config " + quote("config.yaml")
    + ' -n_sample 10000 ')

# STEP2: TRAIN
bash(' onmt_train ' + 
    ' -config ' + quote("config.yaml") )

# STEP3: TRANSLATE
bash (' onmt_translate ' +
' -model ' + quote('save/model_step_50000.pt')+
' -src ' + quote('ParaCrawl.en-pt.pt.subword.test') + 
' -output ' + quote('save/pred_.txt') +
' -gpu 0' +
' -verbose'
)

#STEP4: EVALUATION

bash(' python3 ' + ' compute-bleu.py ' + ' ' + ' ')



