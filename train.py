import argparse
import tensorflow as tf
tf.python.control_flow_ops = tf
from model import *
from TrainingStrategy import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('strategy', nargs='?', help='Default = pair')
    parser.add_argument('model', nargs='?', help='Default = GRU L1diff correct')
    parser.add_argument('--mtl_ratio', nargs='?', type=float)
    parser.add_argument('--nodes1',nargs='?', type=int)
    parser.add_argument('--nodes2', nargs='?', type=int)
    parser.add_argument('--nb_epoch', nargs='?', type=int)
    parser.add_argument('--dropout_rate', nargs='?', type=float)
    parser.add_argument('--batch_size', nargs='?', type=int)
    parser.add_argument('--nb_test', nargs='?', type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')  # debug flag

    args = parser.parse_args()
    args.strategy = args.strategy if args.strategy else 'pair'
    args.model = args.model if args.model else 'GRU L1diff correct'
    args.nodes1 = args.nodes1 if args.nodes1 else 128
    args.nodes2 = args.nodes2 if args.nodes2 else 0
    args.nb_epoch = args.nb_epoch if args.nb_epoch else 200
    args.nb_test = args.nb_test if args.nb_test else 10
    args.dropout_rate = args.dropout_rate if args.dropout_rate else 0.5
    args.batch_size = args.batch_size if args.batch_size else 250
    args.mtl_ratio = args.mtl_ratio if args.mtl_ratio else 0.5
    print args

    strategies = {'pair': IterativeImproveStrategy, 'LM': LanguageModelTrainingStrategy}
    ts = strategies[args.strategy](args)
    if args.strategy == 'LM': args.one_hot_dim = ts.ydim
    model = build_model(args, args.nodes1, args.nodes2, args.dropout_rate, ts.seq_len)
    ts.train(model)
