# from util import top3notes, InputParser
from util import *
import csv

class HistoryWriter(object):
    def __init__(self):
        self.state = [['epoch'], ['loss'], ['val_loss'], ['errCntAvg']]

    def write_history(self, hist, epoch, errCntAvg):
        state = self.state
        state[0].append(epoch)
        state[1].append(round(hist.history['loss'][0], 2))
        state[2].append(round(hist.history['val_loss'][0], 2))
        state[3].append(round(errCntAvg, 2))

class TrainingStrategy(object):
    def __init__(self):
        pass


    def train(self):
        raise NotImplementedError('override this!')


class PairTrainingStrategy(TrainingStrategy):
    pass
    # alg = self.alg
    # if 'LM' in self.alg:
    #     chord2signature = onehot2notes_translator() if 'one-hot' in self.alg else top3notes
    #
    # # M = training melody
    # # m = testing melody
    # # C = training chord progression
    # # c = testing chord progression
    #
    # ip = InputParser(alg)
    # if 'one-hot' in alg:
    #     alg['one-hot-dim'] = y.shape[2]
    #
    # # X = training features
    # # x = validation features (to evaluate val_loss & val_acc)
    # # Y = training ground truth
    # # y = validation ground truth
    # # x_test = testing features (to evaluate unique_idx & norms)
    # if 'pair' in alg:  # pair model can load data first since they're small, only load 1st npy file
    #     nb_test = 100
    #     M, m, C, c, SW, sw_val = load_data(alg, nb_test)
    #     x_test = get_test(alg, m, C)
    #     nb_train = M.shape[0]
    #     seq_len = 128
    # else:
    #     m_val, m_test, c_val, c_test, sw_val, _ = load_testval_data()
    #     x, y = ip.get_XY(m_val, c_val)
    #     x_test = m_test
    #     seq_len = m_test.shape[1]
    #
    # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
    # history = [['epoch'], ['loss'], ['val_loss'], ['errCntAvg']]
    #
    # # history will record the loss of every epoch
    # # since it's too time-consuming to compute the unique_idx and norms,
    # # record and save models after nb_epoch_pred epochs
    #
    # def _get_filename(_alg):
    #     major = 'LM' if 'LM' in _alg else 'pair' if 'pair' in _alg else ''
    #     minor = 'onehot' if 'one-hot' in _alg else 'rand' if 'rand' in _alg else 'L1diff' if 'L1diff' in _alg else 'L1'
    #     rnn = 'RNN' if 'RNN' in _alg else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg else ''
    #     if 'Bidirectional' in _alg: rnn = 'B' + rnn
    #
    #     fn = rnn + '_' + major
    #     if minor:
    #         fn += '_' + minor
    #     fn += '_nodes' + str(nodes1)
    #     return fn
    #
    # filename = _get_filename(alg)
    #
    # for i in range(nb_epoch / nb_epoch_pred):
    #     if 'pair' in alg:  # shuffle negative samples
    #         for j in range(nb_epoch_pred):
    #             epoch = nb_epoch_pred * i + j + 1
    #             sys.stdout.write("Alg=%s, epoch=%d\r" % (alg, epoch))
    #             sys.stdout.flush()
    #             X, Y = ip.get_XY(M, C)
    #             x, y = ip.get_XY(m, c)
    #             hist = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0, validation_data=(x, y))
    #     else:  # in LM we load all 5 npy files
    #         for i in range(1, 6):
    #             M, C, SW = load_train_data(i)
    #             epoch = nb_epoch_pred * (i - 1)
    #             sys.stdout.write("Alg=%s, epoch=%d to %d\r" % (alg, epoch, epoch + nb_epoch_pred))
    #             sys.stdout.flush()
    #             X, Y = ip.get_XY(M, C)
    #             hist = model.fit(X, Y, sample_weight=SW, batch_size=batch_size, nb_epoch=nb_epoch_pred, verbose=0,
    #                              validation_data=(x, y, sw_val))
    #
    #     # testing
    #     pred = np.array(model.predict(x_test))
    #     if 'LM' in alg:
    #         xdim = alg.get('one-hot-dim', 12)
    #         pred = pred.reshape((nb_test, seq_len, xdim))
    #         if seq_len % 16 != 0:
    #             tail = pred[:][:, -seq_len % 16:]
    #             tail = np.average(tail, axis=1).reshape((nb_test, 1, xdim))
    #             tail = np.tile(tail, (1, seq_len % 16, 1))
    #         head = pred[:][:, :seq_len / 16 * 16].reshape((nb_test, 16, seq_len / 16, xdim))
    #         head = np.average(head, axis=1)
    #         head = np.tile(head, (1, 16, 1))
    #         if seq_len % 16 != 0:
    #             predAvg = np.concatenate((head, tail), axis=1)
    #         else:
    #             predAvg = head
    #         y2 = chord2signature(y) if 'one-hot' in alg else y  # use notes representation for y
    #         c_hat = chord2signature(pred)
    #         c_hatAvg = chord2signature(predAvg)
    #         errCntAvg = np.average(np.abs(y2 - c_hat)) * 12
    #         errCntAvgAvg = np.average(np.abs(y2 - c_hatAvg)) * 12
    #         np.save('../pred/' + filename + '.npy', c_hat.astype(int).reshape((nb_test, seq_len, 12)))
    #         np.save('../pred/' + filename + '.npy', c_hatAvg.astype(int).reshape((nb_test, seq_len, 12)))
    #     elif 'pair' in alg:
    #         if 'L1diff' in alg:
    #             pred = pred.reshape((nb_test, nb_train, 128 * 12))
    #             idx = np.argmin(np.sum(np.abs(pred - 0.5), axis=2), axis=1)
    #         else:
    #             pred = pred.reshape((nb_test, nb_train, 128))
    #             idx = np.argmax(np.sum(pred, axis=2), axis=1)
    #         c_hat = C[idx]
    #         # bestN, uniqIdx, norm = print_result(c_hat, c, C, alg, False, 1)
    #         # L1 error
    #         if 'L1' in alg or 'L1diff' in alg:
    #             errCntAvg = np.average(np.abs(c_hat - c)) * 12
    #         # F1 error
    #         elif 'F1' in alg:
    #             np.seterr(divide='ignore', invalid='ignore')  # turn off warning of division by zero
    #             p = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c_hat, 2)
    #             r = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c, 2)
    #             errCntAvg = np.average(np.nan_to_num(2 * p * r / (p + r)))
    #         np.save('../pred/' + filename + '.npy', c_hat.astype(int).reshape((nb_test, 128, 12)))
    #
    #     history = write_history(history, hist, nb_epoch_pred * (i + 1), errCntAvg)
    #     with open('history/' + filename + '.csv', 'w') as csvfile:
    #         csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history)))
    #     print "epoch:", history[0][-1], "train_loss:", history[1][-1], "test_loss:", history[2][-1], "errCntAvg:", \
    #         history[3][-1]
    #
    #     # record & save model
    #     # record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
    #     # save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))


class LanguageModelTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        self.is_onehot = 'one-hot' in alg

        alg = self.alg
        self.chord2signature = get_onehot2chordnotes_transcoder() if 'one-hot' in self.alg else top3notes

        ## Naming Guide:
        # M = training melody
        # m = testing melody
        # C = training chord progression
        # c = testing chord progression
        self.ip = InputParser(self.alg)


        ## Naming Guide
        # X = training features
        # x = validation features (to evaluate val_loss & val_acc)
        # Y = training ground truth
        # y = validation ground truth
        # x_test = testing features (to evaluate unique_idx & norms)
        nb_test = 100
        M, m, C, c, SW, sw_val = load_data(alg, nb_test)

        self.SW, self.sw_val = SW, sw_val
        self.X, self.Y = self.ip.get_XY(M, C)
        self.x, self.y = self.ip.get_XY(m, c)
        self.x_test = m
        self.seq_len = 128
        self.ydim = C.shape[2]


    def getYDim(self):
        return self.ydim


    def train(self, model):
        nodes1 = self.alg['nodes1']
        nodes2 = self.alg['nodes2']
        nb_epoch = self.alg['nb_epoch']
        nb_epoch_pred = self.alg['nb_epoch_pred']
        batch_size = self.alg['batch_size']
        seq_len = self.seq_len
        nb_test = 100  # FIXME: Magic Number!!

        X, Y = self.X, self.Y
        x, y = self.x, self.y
        x_test = self.x_test
        SW, sw_val = self.SW, self.sw_val
        # loaded data

        # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
        history = HistoryWriter()

        # history will record the loss of every epoch
        # since it's too time-consuming to compute the unique_idx and norms,
        # record and save models after nb_epoch_pred epochs

        def _get_filename(_alg):
            major = 'LM' if 'LM' in _alg else 'pair' if 'pair' in _alg else ''
            minor = 'onehot' if 'one-hot' in _alg else 'rand' if 'rand' in _alg else 'L1diff' if 'L1diff' in _alg else 'L1'
            rnn = 'RNN' if 'RNN' in _alg else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg else ''
            if 'Bidirectional' in _alg: rnn = 'B' + rnn

            fn = rnn + '_' + major
            if minor:
                fn += '_' + minor
            fn += '_nodes' + str(nodes1)
            return fn

        filename = _get_filename(self.alg)

        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, Y, sample_weight=SW, batch_size=batch_size, nb_epoch=1, verbose=0,
                             validation_data=(x, y, sw_val))

            # testing
            pred = np.array(model.predict(x_test))

            xdim = self.ydim if self.is_onehot else 12
            pred = pred.reshape((nb_test, seq_len, xdim))
            head = pred[:][:, :seq_len].reshape((nb_test, 16, seq_len / 16, xdim))
            head = np.average(head, axis=1)
            head = np.tile(head, (1, 16, 1))
            predAvg = head

            # signature here refers to theo output feature vector to be used for training
            y2 = self.chord2signature(y) if self.is_onehot else y
            c_hat = self.chord2signature(pred)
            c_hatAvg = self.chord2signature(predAvg)
            errCntAvg = np.average(np.abs(y2 - c_hat)) * 12
            # errCntAvgAvg = np.average(np.abs(y2 - c_hatAvg)) * 12
            np.save('../pred/' + filename + '.npy', c_hat.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + '_avg.npy', c_hatAvg.astype(int).reshape((nb_test, seq_len, 12)))


            # record something
            history.write_history(hist, nb_epoch_pred * (i + 1), errCntAvg)
            with open('history/' + filename + '.csv', 'w') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history.state)))
            print "epoch:", history[0][-1], "train_loss:", history[1][-1], "test_loss:", history[2][-1], "errCntAvg:", \
            history.state[3][-1]

            # record & save model
            # record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
            # save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))