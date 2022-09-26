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
yaml_config_test =  """
# toy_en_de.yaml

## Where the samples will be written
save_data: toy-ende/run/example
## Where the vocab(s) will be written
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt
# Prevent overwriting existing files in the folder
overwrite: False

# Corpus opts:
data:
    corpus_1:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
# Train on a single GPU
world_size: 1
gpu_ranks: [0]

# Where to save the checkpoints
save_model: toy-ende/run/model
save_checkpoint_steps: 500
train_steps: 1000
valid_steps: 500

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

    
        
    
        
        

"""

config = yaml.safe_load(yaml_config_test) #ver o que isto faz
# write yaml_config to a file
with open("toy-ende/config_test.yaml", 'w') as f:
    f.write(yaml_config_test)

#STEP1: BUILD VOCAB
#bash(" onmt_build_vocab " +
#    " -config " + quote("toy-ende/config_test.yaml")
#    + ' -n_sample 1000  ')

# STEP2: TRAIN
#bash(' onmt_train ' + 
#    ' -config ' + quote('toy-ende/config_test.yaml'))
    

# STEP3: TRANSLATE
bash (' onmt_translate ' +
' -model ' + quote('toy-ende/run/model_step_1000.pt')+
' -src ' + quote('toy-ende/src.test.txt') +  
' -output ' + quote('toy-ende/run/pred_1000.txt') +
' -gpu 0' +
' -verbose'
)

#STEP4: INFERENCE
bash(' onmt_release_model ' + 
'-model ' + quote(' model_step.test ') + 
' -output ' + quote(' model_step._release.test '))
