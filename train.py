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
    parser.add_argument(dest='algorithm', metavar='algorithm', nargs='?', default='GRU LM one-hot')
    parser.add_argument(dest='nodes1', nargs='?', type=int, default=64)
    parser.add_argument(dest='nodes2', nargs='?', type=int, default=64)
    parser.add_argument(dest='nb_epoch', nargs='?', type=int, default=200)
    parser.add_argument(dest='nb_epoch_pred', nargs='?', type=int, default=10)

    parser.add_argument(dest='dropout_rate', nargs='?', type=float, default=0.2)
    parser.add_argument(dest='batch_size', nargs='?', type=int, default=500)
    parser.add_argument(dest='nb_test', nargs='?', type=int, default=5000)
    args = parser.parse_args()

    alg = parse_algorithm(args.algorithm)
    alg.update(vars(args))
    nodes1, nodes2,  = args.nodes1, args.nodes2
    nb_epoch, nb_epoch_pred, nb_test = args.nb_epoch, args.nb_epoch_pred, args.nb_test
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size

    # I am thinking it should be like this
    if 'pair' in alg:
        ts = PairTrainingStrategy(alg)
    elif 'LM' in alg:
        ts = LanguageModelTrainingStrategy(alg)
    else:
        raise ValueError('Please specify a valid training strategy!')

    if 'one-hot' in alg:
        alg['one-hot-dim'] = ts.ydim
    model = build_model(alg, nodes1, nodes2, dropout_rate, ts.seq_len)
    ts.train(model)