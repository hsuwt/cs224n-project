<<<<<<< HEAD
# from util import top3notes, InputParser
from util import *
import csv

class HistoryWriterPair(object):
    def __init__(self):
        self.state = [['epoch'], ['loss'], ['val_loss'], ['errCntAvg'],['uniqIdx'], ['norm']]
        
    def write_history(self, hist, epoch, errCntAvg, uniqIdx, norm):
        state = self.state
        state[0].append(epoch)
        state[1].append(round(hist.history['loss'][0], 2))
        state[2].append(round(hist.history['val_loss'][0], 2))
        state[3].append(round(errCntAvg, 2))
        state[4].append(uniqIdx)
        state[5].append(norm)        

class HistoryWriterLM(object):
    def __init__(self):
        self.state = [['epoch'], ['train1'], ['train12'], ['val1'], ['val12'], ['err1'], ['err12'], ['err1Avg'], ['err12Avg']]

    def write_history(self, hist, epoch, errCntAvg):
        state = self.state
        state[0].append(epoch)
        state[1].append(round(hist.history['one-hot_loss'][0], 2))
        state[2].append(round(hist.history['chroma_loss'][0], 2))
        state[3].append(round(hist.history['val_one-hot_loss'][0], 2))
        state[4].append(round(hist.history['val_chroma_loss'][0], 2))
        state[5].append(round(errCntAvg[0], 2))
        state[6].append(round(errCntAvg[1], 2))
        state[7].append(round(errCntAvg[2], 2))
        state[8].append(round(errCntAvg[3], 2))

class TrainingStrategy(object):
    def __init__(self):
        pass


    def train(self):
        raise NotImplementedError('override this!')


    def get_filename(self, _alg):
        major = 'LM' if 'LM' in _alg else 'pair' if 'pair' in _alg else 'attention' if 'attention' in _alg else ''
        minor = 'onehot' if 'one-hot' in _alg else 'rand' if 'rand' in _alg else 'L1diff' if 'L1diff' in _alg else ''
        rnn = 'RNN' if 'RNN' in _alg else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg else ''
        if 'Bidirectional' in _alg: rnn = 'B' + rnn

        fn = rnn + '_' + major
        if minor:
            fn += '_' + minor
        if 'sample-biased' in _alg:
            fn += '_' + 'sb'
        fn += '_nodes' + str(_alg["nodes1"])
        if 'mtl_ratio' in _alg:
            fn += '_' + str(_alg['mtl_ratio'])
        return fn


class PairTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        self.ip = InputParser(alg)
        ## Naming Guide
        # X = training features
        # x = validation features (to evaluate val_loss & val_acc)
        # Y = training ground truth
        # y = validation ground truth
        # x_test = testing features (to evaluate unique_idx & norms)
        self.nb_test = 100
        M, m, C, c, SW, sw_val = load_data(alg, self.nb_test)

        self.SW, self.sw_val = SW, sw_val
        self.X, self.Y = self.ip.get_XY(M, C)
        self.x, self.y = self.ip.get_XY(m, c)
        self.x_test = get_test(alg, m, C)
        self.c, self.C = c, C
        self.seq_len = 128
        self.nb_train = M.shape[0]

    def train(self, model):
        nodes1 = self.alg['nodes1']
        nodes2 = self.alg['nodes2']
        nb_epoch = self.alg['nb_epoch']
        batch_size = self.alg['batch_size']
        seq_len = self.seq_len

        alg = self.alg
        X, Y = self.X, self.Y
        x, y = self.x, self.y
        x_test = self.x_test
        c, C = self.c, self.C
        nb_train, nb_test = self.nb_train, self.nb_test
        SW, sw_val = self.SW, self.sw_val
        # loaded data

        # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
        history = HistoryWriterPair()

        # history will record the loss of every epoch
        # since it's too time-consuming to compute the unique_idx and norms,
        # record and save models after nb_epoch_pred epochs

        filename = self.get_filename(self.alg)

        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0,
                             validation_data=(x, y))

            # testing
            pred = np.array(model.predict(x_test))
            if 'L1diff' in alg:
                pred = pred.reshape((nb_test, nb_train, 128 * 12))
                idx = np.argmin(np.sum(np.abs(pred - 0.5), axis=2), axis=1)
            else:
                pred = pred.reshape((nb_test, nb_train, 128))
                idx = np.argmax(np.sum(pred, axis=2), axis=1)
            c_hat = C[idx]
            bestN, uniqIdx, norm = print_result(c_hat, c, C, alg, False, 1)
            # L1 error
            if 'L1' in alg or 'L1diff' in alg:
                errCntAvg = np.average(np.abs(c_hat - c)) * 12
                # F1 error
            elif 'F1' in alg:
                np.seterr(divide='ignore', invalid='ignore')  # turn off warning of division by zero
                p = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c_hat, 2)
                r = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c, 2)
                errCntAvg = np.average(np.nan_to_num(2 * p * r / (p + r)))
            np.save('../pred/' + filename + '.npy', c_hat.astype(int).reshape((nb_test, 128, 12)))

            # record something
            history.write_history(hist, i+1, errCntAvg,uniqIdx, norm )
            with open('history/' + filename + '.csv', 'w') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history.state)))
            print "epoch:", history.state[0][-1], "train_loss:", history.state[1][-1], \
                "test_loss:", history.state[2][-1], "errCntAvg:", \
            history.state[3][-1]


class LanguageModelTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        alg = self.alg
        self.chord2signatureOnehot = get_onehot2chordnotes_transcoder()
        self.chord2signatureChroma = top3notes

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
        self.X, self.YChroma, self.YOnehot = self.ip.get_XY(M, C)
        self.x, self.yChroma, self.yOnehot = self.ip.get_XY(m, c)
        self.x_test = m
        self.seq_len = 128
        self.ydim = self.YOnehot.shape[2]

    def getYDim(self):
        return self.ydim


    def train(self, model):
        nodes1 = self.alg['nodes1']
        nodes2 = self.alg['nodes2']
        nb_epoch = self.alg['nb_epoch']
        batch_size = self.alg['batch_size']
        seq_len = self.seq_len
        nb_test = 100  # FIXME: Magic Number!!

        X, YChroma, YOnehot = self.X, self.YChroma, self.YOnehot
        x, yChroma, yOnehot = self.x, self.yChroma, self.yOnehot
        x_test = self.x_test
        SW, sw_val = self.SW, self.sw_val
        # loaded data

        # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
        history = HistoryWriterLM()

        # history will record the loss of every epoch
        # since it's too time-consuming to compute the unique_idx and norms,
        # record and save models after nb_epoch_pred epochs

        filename = self.get_filename(self.alg)

        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, {'one-hot':YOnehot, 'chroma':YChroma}, sample_weight={'one-hot':SW, 'chroma':SW}, batch_size=batch_size, nb_epoch=1, verbose=0,
                             validation_data=(x, {'one-hot':yOnehot, 'chroma':yChroma}, {'one-hot':sw_val, 'chroma':sw_val}))
            # testing
            print nb_test

            predOnehot, predChroma = model.predict(x_test)
            predOnehot = np.array(predOnehot).reshape((nb_test, seq_len, self.ydim))
            predChroma = np.array(predChroma).reshape((nb_test, seq_len, 12))
            predOnehotAvg = (predOnehot + 0.0).reshape((nb_test, seq_len / 8, 8, self.ydim))
            predChromaAvg = (predChroma + 0.0).reshape((nb_test, seq_len / 8, 8, 12))
            predOnehotAvg = np.average(predOnehotAvg, axis=2)
            predChromaAvg = np.average(predChromaAvg, axis=2)
            predOnehotAvg = np.repeat(predOnehotAvg, 8, axis=1)
            predChromaAvg = np.repeat(predChromaAvg, 8, axis=1)

            # signature here refers to theo output feature vector to be used for training
            c_hatOnehot    = self.chord2signatureOnehot(predOnehot)
            c_hatChroma    = self.chord2signatureChroma(predChroma)
            c_hatOnehotAvg = self.chord2signatureOnehot(predOnehotAvg)
            c_hatChromaAvg = self.chord2signatureChroma(predChromaAvg)
            errCntAvgOnehot    = np.average(np.abs(yChroma - c_hatOnehot)) * 12
            errCntAvgChroma    = np.average(np.abs(yChroma - c_hatChroma)) * 12
            errCntAvgOnehotAvg = np.average(np.abs(yChroma - c_hatOnehotAvg)) * 12
            errCntAvgChromaAvg = np.average(np.abs(yChroma - c_hatChromaAvg)) * 12
            np.save('../pred/' + filename + 'Onehot.npy', c_hatOnehot.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'Chroma.npy', c_hatChroma.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'OnehotAvg.npy', c_hatOnehotAvg.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'ChromaAvg.npy', c_hatChromaAvg.astype(int).reshape((nb_test, seq_len, 12)))
            errCntAvg = [errCntAvgOnehot, errCntAvgChroma, errCntAvgOnehotAvg, errCntAvgChromaAvg]
            # record something
            history.write_history(hist, i+1, errCntAvg)
            with open('history/' + filename + '.csv', 'w') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history.state)))
            print "epoch:", history.state[0][-1], "train1:", history.state[1][-1], "train12:", history.state[2][-1], \
            "val1:", history.state[3][-1], "val12:", history.state[4][-1], \
            "err1:", history.state[5][-1], "err12:", history.state[6][-1], \
            "err1Avg:", history.state[7][-1], "err12Avg:", history.state[8][-1]

            # record & save model
            # record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
            # save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))


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

=======
# from util import top3notes, InputParser
from util import *
import csv

class HistoryWriterPair(object):
    def __init__(self):
        self.state = [['epoch'], ['loss1'], ['loss12'],['val_loss1'], ['val_loss12'], ['errCntAvg1'], ['errCntAvg12'],['uniqIdx1'],['uniqIdx12'], ['norm1'], ['norm12']]
        
    def write_history(self, hist, epoch, errCntAvg, uniqIdx, norm):
        state = self.state
        state[0].append(epoch)
        state[1].append(round(hist.history['one-hot_loss'][0], 2))
        state[2].append(round(hist.history['chroma_loss'][0], 2))
        state[3].append(round(hist.history['val_one-hot_loss'][0], 2))
        state[4].append(round(hist.history['val_chroma_loss'][0], 2))
        state[5].append(round(errCntAvg[0], 2))
        state[6].append(round(errCntAvg[1], 2))
        state[7].append(uniqIdx[0])
        state[8].append(uniqIdx[1])
        state[9].append(norm[0])    
        state[10].append(norm[1]) 

class HistoryWriterLM(object):
    def __init__(self):
        self.state = [['epoch'], ['train1'], ['train12'], ['val1'], ['val12'], ['err1'], ['err12'], ['err1Avg'], ['err12Avg']]

    def write_history(self, hist, epoch, errCntAvg):
        state = self.state
        state[0].append(epoch)
        state[1].append(round(hist.history['one-hot_loss'][0], 2))
        state[2].append(round(hist.history['chroma_loss'][0], 2))
        state[3].append(round(hist.history['val_one-hot_loss'][0], 2))
        state[4].append(round(hist.history['val_chroma_loss'][0], 2))
        state[5].append(round(errCntAvg[0], 2))
        state[6].append(round(errCntAvg[1], 2))
        state[7].append(round(errCntAvg[2], 2))
        state[8].append(round(errCntAvg[3], 2))


class TrainingStrategy(object):
    def __init__(self):
        pass


    def train(self):
        raise NotImplementedError('override this!')


    def get_filename(self, _alg):
        major = 'LM' if 'LM' in _alg else 'pair' if 'pair' in _alg else 'attention' if 'attention' in _alg else ''
        minor = 'onehot' if 'one-hot' in _alg else 'rand' if 'rand' in _alg else 'L1diff' if 'L1diff' in _alg else ''
        rnn = 'RNN' if 'RNN' in _alg else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg else ''
        if 'Bidirectional' in _alg: rnn = 'B' + rnn

        fn = rnn + '_' + major
        if minor:
            fn += '_' + minor
        if 'sample-biased' in _alg:
            fn += '_' + 'sb'
        fn += '_nodes' + str(_alg["nodes1"])
        if 'mtl_ratio' in _alg:
            fn += '_' + str(_alg['mtl_ratio'])
        return fn


class PairTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        self.ip = InputParser(alg)
        ## Naming Guide
        # X = training features
        # x = validation features (to evaluate val_loss & val_acc)
        # Y = training ground truth
        # y = validation ground truth
        # x_test = testing features (to evaluate unique_idx & norms)
        self.nb_test = 100
        M, m, C, c, SW, sw_val = load_data(alg, self.nb_test)

        self.SW, self.sw_val = SW, sw_val
        self.X, self.YChroma, self.YOnehot = self.ip.get_XY(M, C)
        self.x, self.yChroma, self.yOnehot = self.ip.get_XY(m, c)
        self.x_test = get_test(alg, m, C)
        self.c, self.C = c, C
        self.seq_len = 128
        self.nb_train = M.shape[0]

    def train(self, model):
        nodes1 = self.alg['nodes1']
        nodes2 = self.alg['nodes2']
        nb_epoch = self.alg['nb_epoch']
        batch_size = self.alg['batch_size']
        seq_len = self.seq_len

        alg = self.alg
        X, YChroma, YOnehot = self.X, self.YChroma, self.YOnehot
        x, yChroma, yOnehot = self.x, self.yChroma, self.yOnehot
        x_test = self.x_test
        c, C = self.c, self.C
        nb_train, nb_test = self.nb_train, self.nb_test
        SW, sw_val = self.SW, self.sw_val
        # loaded data

        # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
        history = HistoryWriterPair()

        # history will record the loss of every epoch
        # since it's too time-consuming to compute the unique_idx and norms,
        # record and save models after nb_epoch_pred epochs

        filename = self.get_filename(self.alg)

        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, {'one-hot': YOnehot, 'chroma': YChroma}, batch_size=batch_size, nb_epoch=1, verbose=0,
                             validation_data=(x, {'one-hot': yOnehot, 'chroma': yChroma}))
      
            predOnehot, predChroma = model.predict(x_test)

            # testing
            predOnehot, predChroma = model.predict(x_test)            
            if 'L1diff' in alg:
                predChroma = np.array(predChroma).reshape((nb_test, nb_train, 128 * 12))
                predChroma = predChroma.reshape((nb_test, nb_train, 128 * 12))
                idxChroma = np.argmin(np.sum(np.abs(predChroma - 0.5), axis=2), axis=1)
                predOnehot = np.array(predOnehot).reshape((nb_test, nb_train, 128 * 119))
                predOnehot = predOnehot.reshape((nb_test, nb_train, 128 * 119)) 
                idxOnehot = np.argmin(np.sum(np.abs(predOnehot - 0.5), axis=2), axis=1)  
            else:
                raise ValueError("Mtl only takes L1diff")
                #pred = pred.reshape((nb_test, nb_train, 128))
                #idx = np.argmax(np.sum(pred, axis=2), axis=1)
            c_hatChroma = C[idxChroma]
            c_hatOnehot = C[idxOnehot]
            bestNChroma, uniqIdxChroma, normChroma = print_result(c_hatChroma, c, C, alg, False, 1)
            bestNOnehot, uniqIdxOnehot, normOnehot = print_result(c_hatOnehot, c, C, alg, False, 1)
            # L1 error
            if 'L1' in alg or 'L1diff' in alg:
                errCntAvgChroma = np.average(np.abs(c_hatChroma - c)) * 12
                errCntAvgOnehot = np.average(np.abs(c_hatOnehot - c)) * 12                
                # F1 error
            elif 'F1' in alg:
                np.seterr(divide='ignore', invalid='ignore')  # turn off warning of division by zero
                p = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c_hat, 2)
                r = np.sum(np.logical_and(c, c_hat), 2) / np.sum(c, 2)
                errCntAvg = np.average(np.nan_to_num(2 * p * r / (p + r)))
            np.save('../pred/' + filename + 'Chroma.npy', c_hatChroma.astype(int).reshape((nb_test, 128, 12)))
            np.save('../pred/' + filename + 'Onehot.npy', c_hatOnehot.astype(int).reshape((nb_test, 128, 12)))

            # record something
            history.write_history(hist, i+1, [errCntAvgOnehot, errCntAvgChroma],[uniqIdxOnehot, uniqIdxChroma] , \
                                  [ normOnehot, normChroma] )
            with open('history/' + filename + '.csv', 'w') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history.state)))
            print "epoch:", history.state[0][-1], "train_loss1:", history.state[1][-1],"train_loss12:", history.state[2][-1], \
                "test_loss1:", history.state[3][-1], "test_loss12:", history.state[4][-1], "errCntAvg1:", \
            history.state[5][-1], "errCntAvg12:", history.state[6][-1]


class LanguageModelTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        alg = self.alg
        self.chord2signatureOnehot = get_onehot2chordnotes_transcoder()
        self.chord2signatureChroma = top3notes

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
        self.X, self.YChroma, self.YOnehot = self.ip.get_XY(M, C)
        self.x, self.yChroma, self.yOnehot = self.ip.get_XY(m, c)
        self.x_test = m
        self.seq_len = 128
        self.ydim = self.YOnehot.shape[2]

    def getYDim(self):
        return self.ydim


    def train(self, model):
        nodes1 = self.alg['nodes1']
        nodes2 = self.alg['nodes2']
        nb_epoch = self.alg['nb_epoch']
        batch_size = self.alg['batch_size']
        seq_len = self.seq_len
        nb_test = 100  # FIXME: Magic Number!!

        X, YChroma, YOnehot = self.X, self.YChroma, self.YOnehot
        x, yChroma, yOnehot = self.x, self.yChroma, self.yOnehot
        x_test = self.x_test
        SW, sw_val = self.SW, self.sw_val
        # loaded data

        # model = build_model(alg, nodes1, nodes2, dropout_rate, seq_len)
        history = HistoryWriterLM()

        # history will record the loss of every epoch
        # since it's too time-consuming to compute the unique_idx and norms,
        # record and save models after nb_epoch_pred epochs

        filename = self.get_filename(self.alg)

        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, {'one-hot': YOnehot, 'chroma': YChroma}, sample_weight={'one-hot': SW, 'chroma': SW}, batch_size=batch_size, nb_epoch=1, verbose=0,
                             validation_data=(x, {'one-hot': yOnehot, 'chroma': yChroma}, {'one-hot': sw_val, 'chroma': sw_val}))
            # testing
            predOnehot, predChroma = model.predict(x_test)
            predOnehot = np.array(predOnehot).reshape((nb_test, seq_len, self.ydim))
            predChroma = np.array(predChroma).reshape((nb_test, seq_len, 12))
            predOnehotAvg = (predOnehot + 0.0).reshape((nb_test, seq_len / 8, 8, self.ydim))
            predChromaAvg = (predChroma + 0.0).reshape((nb_test, seq_len / 8, 8, 12))
            predOnehotAvg = np.average(predOnehotAvg, axis=2)
            predChromaAvg = np.average(predChromaAvg, axis=2)
            predOnehotAvg = np.repeat(predOnehotAvg, 8, axis=1)
            predChromaAvg = np.repeat(predChromaAvg, 8, axis=1)

            # signature here refers to theo output feature vector to be used for training
            yOnehot12      = self.chord2signatureOnehot(yOnehot)
            c_hatOnehot    = self.chord2signatureOnehot(predOnehot)
            c_hatChroma    = self.chord2signatureChroma(predChroma)
            c_hatOnehotAvg = self.chord2signatureOnehot(predOnehotAvg)
            c_hatChromaAvg = self.chord2signatureChroma(predChromaAvg)
            errCntAvgOnehot    = np.average(np.abs(yOnehot12 - c_hatOnehot)) * 12
            errCntAvgChroma    = np.average(np.abs(yChroma   - c_hatChroma)) * 12
            errCntAvgOnehotAvg = np.average(np.abs(yOnehot12 - c_hatOnehotAvg)) * 12
            errCntAvgChromaAvg = np.average(np.abs(yChroma   - c_hatChromaAvg)) * 12
            np.save('../pred/' + filename + 'Onehot.npy', c_hatOnehot.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'Chroma.npy', c_hatChroma.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'OnehotAvg.npy', c_hatOnehotAvg.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'ChromaAvg.npy', c_hatChromaAvg.astype(int).reshape((nb_test, seq_len, 12)))
            errCntAvg = [errCntAvgOnehot, errCntAvgChroma, errCntAvgOnehotAvg, errCntAvgChromaAvg]
            # record something
            history.write_history(hist, i+1, errCntAvg)
            with open('history/' + filename + '.csv', 'w') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history.state)))
            print "epoch:", history.state[0][-1], "train1:", history.state[1][-1], "train12:", history.state[2][-1], \
            "val1:", history.state[3][-1], "val12:", history.state[4][-1], \
            "err1:", history.state[5][-1], "err12:", history.state[6][-1], \
            "err1Avg:", history.state[7][-1], "err12Avg:", history.state[8][-1]

            # record & save model
            # record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
            # save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))


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

>>>>>>> 6c8226b8f3fcefb4c7e819683cd59c56973e1269
