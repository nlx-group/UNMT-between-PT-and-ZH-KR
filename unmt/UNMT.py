from logging import root
import os
import argparse
import subprocess
import sys
from shlex import quote
#give path where this is
#give vecmap path
ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = ROOT + '/THIRD_PARTY'
VECMAP = THIRD_PARTY + '/vecmap'
WORDVEC = THIRD_PARTY + '/word2vec/wordvec'
TOKENIZER = THIRD_PARTY + '/tokenizer'
os.environ['PATH'] =  ':'.join([os.environ['PATH']])
PHRASE2VEC = THIRD_PARTY + '/phrase2vec'
UNDREAMT = THIRD_PARTY + '/undreamt'
FASTTEXT = THIRD_PARTY + '/fastText' 
SUBWORD_NMT = THIRD_PARTY + '/subword-nmt'
TRAINING = ROOT + '/training'


def bash(command):
    subprocess.run(['bash', '-c', command])
 
def subword(args):
    root =  args.working 
    os.mkdir(root)
    root += '/step1'
    os.mkdir(root)
    #pass the python script grab the txt file
    for part, corpus in (('src', args.src),('trg', args.trg)):
        #corpus = corpus + '.' + part 
        bash(' python3 ' + quote(SUBWORD_NMT + '/learn_bpe.py')+
        ' < ' + quote(corpus) +
        ' -s 2000' +
        ' > ' + quote(root + '/subword-learn.txt' + '.' + part))

def apply_subword(args):
    root = args.working + '/step2'
    os.mkdir(root)
    for part, corpus in (('src', args.src),('trg', args.trg)):
        bash(' python3 ' + quote(SUBWORD_NMT + '/subword_nmt/apply_bpe.py') +
        ' -c' + quote(args.working + '/step1/subword-learn.txt' + '.' + part) +
        ' < ' + quote(corpus) +
        ' > '+ quote(root + '/subword_bpe.txt' + '.' + part))


# step3 train word2vec
def word2vec(args):
    root = args.working + '/step3'
    os.mkdir(root)
    for part in ('src','trg'): 
        corpus = args.working + '/step2/subword_bpe.txt' + '.' + part 
        bash(' python3 ' + quote(WORDVEC + '/vec.py') +
        ' --src ' + quote(corpus) +
        ' --size ' + str(args.size) +
        ' --window ' + str(args.window) +
        ' --lr ' + str(args.lr) +
        ' --iter ' + str(args.iter) +
        ' --save ' + quote(root  + '/bin.' + part) + #saved file as bin
        ' --vec '  + quote(root  + '/vec.' + part) #saves file as vec
        )

#step 4 train vecmap
#add cuda support
def vecmap(args):
    root = args.working + '/step4'
    os.makedirs(root)
    bash(' python3 ' + quote(VECMAP + '/map_embeddings.py') +
    ' --' + args.mode + ' -v' +
    ' ' + quote(args.working + '/step3/vec.src') +
    ' ' + quote(args.working + '/step3/vec.trg') +
    ' ' + quote(root + '/emb.src') +
    ' ' + quote(root + '/emb.trg'))
    

# step3 Train NMT
def undreamt(args):
    root = args.working + '/step5'
    os.mkdir(root) 
    src = args.working + '/step2/subword_bpe.txt.src'
    trg = args.working + '/step2/subword_bpe.txt.trg'

    bash(' python3 ' + quote(UNDREAMT + '/train.py') +
    ' --src ' + quote(src) +
    ' --trg ' + quote(trg) +
    ' --src_embeddings ' + quote(args.working + '/step4' + '/emb.src') +
    ' --trg_embeddings ' + quote(args.working + '/step4' + '/emb.trg') +
    ' --save Model' +
    ' --cutoff 40000')

def main():
    #add arguments
    parser = argparse.ArgumentParser(description="Create vecmap")
    parser.add_argument('--src', metavar='STR', required=True, help='Source language corpus')
    parser.add_argument('--src_lang', metavar='STR', required=True, help='Source language')
    parser.add_argument('--trg', metavar='STR', required=True,  help='Target language corpus')
    parser.add_argument('--trg_lang', metavar='STR', required=True,  help='Target language corpus')
    parser.add_argument('--working', help='Creates the folder to save all')
    #parser.add_argument('trg_output', help='the output target embeddings')

    #tokenize --src_output
    tokenize_group = parser.add_argument_group(description="Create tokeinzer")
    tokenize_group.add_argument('--src_output')

    word2vec_group = parser.add_argument_group(description="Create word2vec")
    word2vec_group.add_argument('--size', metavar='N', type=int, default=300, help='Dimensionality of the phrase embeddings (defaults to 300)')
    word2vec_group.add_argument('--window', metavar='N', type=int, default=5, help='Max skip length between words (defauls to 5)')
    word2vec_group.add_argument('--lr', metavar='N', type=float, default=0.4, help='Learning rate')
    word2vec_group.add_argument('--negative', metavar='N', type=int, default=10, help='Number of negative examples (defaults to 10)')
    word2vec_group.add_argument('--iter', metavar='N', type=int, default=5, help='Number of training epochs (defaults to 5)')
    word2vec_group.add_argument('--save', help='output fild')
    word2vec_group.add_argument('--vec')

    #subword_nmt
    subwordnmt_group = parser.add_argument_group(description="Create subword")
    subwordnmt_group.add_argument('-s', default=32000)
    #vecmap
    vecmap_group = parser.add_argument_group('Step 4', 'Embedding mapping')
    vecmap_group.add_argument('--mode', choices=['identical', 'unsupervised'], default='unsupervised', help='VecMap mode (defaults to identical)')

    #undreamt
    undreamt_group = parser.add_argument_group(description="Create undreamt")
    undreamt_group.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    undreamt_group.add_argument('--cutoff', default=40000)


    args = parser.parse_args()
    subword(args)
    apply_subword(args)
    word2vec(args)
    vecmap(args)
    undreamt(args)

if __name__ == '__main__':
    main()
