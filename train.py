import sys
from util import *
from model import *
import time
import argparse
import tensorflow as tf
#from genMIDI import *
tf.python.control_flow_ops = tf

from TrainingStrategy import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(dest='algorithm', metavar='algorithm', nargs='?', default='GRU LM')
    parser.add_argument(dest='nodes1', nargs='?', type=int, default=128)
    parser.add_argument(dest='nodes2', nargs='?', type=int, default=128)
    parser.add_argument(dest='nb_epoch', nargs='?', type=int, default=200)
    parser.add_argument(dest='dropout_rate', nargs='?', type=float, default=0.5)
    parser.add_argument(dest='batch_size', nargs='?', type=int, default=250)
    parser.add_argument(dest='nb_test', nargs='?', type=int, default=100)
    parser.add_argument(dest='mtl_ratio', nargs='?', type=int, default=0)
    args = parser.parse_args()
    alg = parse_algorithm(args.algorithm)
    alg.update(vars(args))
    nodes1, nodes2,  = args.nodes1, args.nodes2
    nb_epoch, nb_test = args.nb_epoch, args.nb_test
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    mtl_ratio = args.mtl_ratio

    # I am thinking it should be like this
    if 'pair' in alg:
        ts = PairTrainingStrategy(alg)
    elif 'LM' in alg:
        ts = LanguageModelTrainingStrategy(alg)
    else:
        raise ValueError('Please specify a valid training strategy!')

    
    if 'LM' in alg:
        alg['one-hot-dim'] = ts.ydim
        
        for i in range(11):
            alg['mtl_ratio'] = 0.1 * i
            model = build_model(alg, nodes1, nodes2, dropout_rate, ts.seq_len)
            ts.train(model)
    else:
        model = build_model(alg, nodes1, nodes2, dropout_rate, ts.seq_len)
        ts.train(model)        
