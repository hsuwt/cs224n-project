from util import *
from model import *
import time
import argparse
import tensorflow as tf
#from genMIDI import *
tf.python.control_flow_ops = tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(dest='algorithm', metavar='algorithm', nargs='?', default='GRU LM one-hot')
    parser.add_argument(dest='nodes1', nargs='?', type=int, default=512)
    parser.add_argument(dest='nodes2', nargs='?', type=int, default=64)
    parser.add_argument(dest='nb_epoch', nargs='?', type=int, default=200)
    parser.add_argument(dest='nb_epoch_pred', nargs='?', type=int, default=10)

    parser.add_argument(dest='dropout_rate', nargs='?', type=float, default=0.2)
    parser.add_argument(dest='batch_size', nargs='?', type=int, default=500)
    parser.add_argument(dest='nb_test', nargs='?', type=int, default=5000)
    args = parser.parse_args()

    alg = parse_algorithm(args.algorithm)
    nodes1, nodes2,  = args.nodes1, args.nodes2
    nb_epoch, nb_epoch_pred, nb_test = args.nb_epoch, args.nb_epoch_pred, args.nb_test
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size

    if 'LM' in alg:
        chord2signature = onehot2notes_translator() if 'one-hot' in alg else top3notes

    # M = training melody
    # m = testing melody
    # C = training chord progression
    # c = testing chord progression

    m_val, m_test, c_val, c_test, sw_val, _ = load_testval_data()
    ip = InputParser(alg)
    x, y = ip.get_XY(m_val, c_val)
    if 'one-hot' in alg:
        alg['one-hot-dim'] = y.shape[2]

    # X = training features
    # x = validation features (to evaluate val_loss & val_acc)
    # Y = training ground truth
    # y = validation ground truth
    # x_test = testing features (to evaluate unique_idx & norms)
    if 'pair' in alg: # pair model can load data first since they're small, only load 1st npy file
        M, C, SW = load_train_data(1)    
        x_test = get_test(alg, m_test, C)
    else:
        x_test = m_test


    nb_train = M.shape[0]
    seq_len = m_test.shape[1]
    model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
    history = [['epoch'], ['loss'], ['val_loss'], ['errCntAvg']]
    # history will record the loss of every epoch
    # since it's too time-consuming to compute the unique_idx and norms,
    # record and save models after nb_epoch_pred epochs

    def _get_filename(_alg):
        major = 'LM' if 'LM' in _alg else 'pair' if 'pair' in _alg else ''
        minor = 'onehot' if 'one-hot' in _alg else 'rand' if 'rand' in _alg else 'L1diff' if 'L1diff' in _alg else 'L1'
        rnn = 'RNN' if 'RNN' in _alg else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg else ''
        if 'Bidirectional' in _alg: rnn = 'B'+rnn

        fn = rnn + '_' + major
        if minor:
            fn += '_' + minor
        fn += '_nodes' + str(nodes1) + '.csv'
        return fn

    filename = _get_filename(alg)

    for i in range(nb_epoch/nb_epoch_pred):
        if 'pair' in alg: #shuffle negative samples
            for j in range(nb_epoch_pred):
                epoch = nb_epoch_pred*i+j+1
                sys.stdout.write("Alg=%s, epoch=%d\r" % (alg, epoch))
                sys.stdout.flush()
                X, Y = ip.get_XY(M, C)
                x, y = ip.get_XY(m, c)
                hist = model.fit(X, Y, sample_weight=SW, batch_size=batch_size, nb_epoch=1, verbose=0, validation_data=(x, y, sw))
        else: # in LM we load all 5 npy files
            for i in range(1,6):
                M, C, SW = load_train_data(i)            
                epoch = nb_epoch_pred*i+j+1
                sys.stdout.write("Alg=%s, epoch=%d to %d\r" % (alg, epoch, epoch+nb_epoch_pred))
                sys.stdout.flush()   
                X, Y = ip.get_XY(M, C)
                hist = model.fit(X, Y, sample_weight=SW, batch_size=batch_size, nb_epoch=nb_epoch_pred, verbose=0, validation_data=(x, y, sw))
            

        # testing
        pred = np.array(model.predict(x_test))
        if 'LM' in alg:
            xdim = alg.get('one-hot-dim', 12)
            pred = pred.reshape((nb_test, seq_len, xdim))
            if seq_len%16 != 0:
                tail = pred[:][:, -seq_len%16:]
                tail = np.average(tail, axis=1).reshape((nb_test, 1, xdim))
                tail = np.tile(tail, (1, seq_len%16, 1))
            head = pred[:][:, :seq_len/16*16].reshape((nb_test, 16, seq_len/16, xdim))
            head = np.average(head, axis=1)
            head = np.tile(head, (1, 16, 1))
            if seq_len%16 != 0:
                pred = np.concatenate((head, tail), axis=1)
            y2 = chord2signature(y) if 'one-hot' in alg else y  # use notes representation for y
            c_hat = chord2signature(pred)
            errCntAvg = np.average(np.abs(y2 - c_hat)) * 12
            with open('pred/' + filename, 'w') as f:
                np.savetxt(f, c_hat.reshape((nb_test*seq_len, 12)), delimiter=',', fmt="%d")
        elif 'pair' in alg:
            if 'L1diff' in alg:
                pred = pred.reshape((nb_test, nb_train, 128 * 12))
                idx = np.argmin(np.sum(np.abs(pred - 0.5), axis=2), axis=1)
            else:
                pred = pred.reshape((nb_test, nb_train, 128))
                idx = np.argmax(np.sum(pred, axis=2), axis=1)
            c_hat = C[idx]
            #bestN, uniqIdx, norm = print_result(c_hat, c, C, alg, False, 1)
            # L1 error
            if 'L1' in alg or 'L1diff' in alg:
                errCntAvg = np.average(np.abs(c_hat - c)) * 12
            # F1 error
            elif 'F1' in alg:
                np.seterr(divide='ignore', invalid='ignore') # turn off warning of division by zero
                p = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c_hat, 2)
                r = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c, 2)
                errCntAvg = np.average(np.nan_to_num(2*p*r/(p+r)))
            with open('pred/' + filename, 'w') as f:
                np.savetxt(f, c_hat.astype(int).reshape((nb_test*128, 12)), delimiter=',', fmt="%d")

        history = write_history(history, hist, nb_epoch_pred * (i+1), errCntAvg)
        with open('history/' + filename, 'w') as csvfile:
            csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history)))
        print "epoch:", history[0][-1], "train_loss:", history[1][-1], "test_loss:", history[2][-1], "errCntAvg:", history[3][-1]

        # record & save model
        # record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
        # save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))

