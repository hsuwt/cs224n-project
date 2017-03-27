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
        self.state = [['epoch'], ['train1'], ['train12'], ['val1'], ['val12'], ['err1'], ['err12'], ['errEn'], ['err1Avg'], ['err12Avg'], ['errEnAvg']]

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
        state[9].append(round(errCntAvg[4], 2))
        state[10].append(round(errCntAvg[5], 2))


class TrainingStrategy(object):
    def __init__(self):
        pass


    def train(self):
        raise NotImplementedError('override this!')


    def get_filename(self, _alg):
        major = _alg.strategy
        minor = 'onehot' if 'one-hot' in _alg.model else  'L1diff' if 'L1diff' in _alg.model else ''
        rnn = 'RNN' if 'RNN' in _alg.model else "GRU" if "GRU" in _alg else "LSTM" if "LSTM" in _alg.model else ''
        if 'Bidirectional' in _alg.model: rnn = 'B' + rnn

        fn = rnn + '_' + major
        if minor:
            fn += '_' + minor
        if 'sample-biased' in _alg.model:
            fn += '_' + 'sb'
        fn += '_nodes' + str(_alg.nodes1)
        if 'mtl_ratio' in _alg:
            fn += '_' + str(_alg.mtl_ratio)
        return fn


class PairTrainingStrategy(TrainingStrategy):
    def __init__(self, alg):
        self.alg = alg
        self.ip = PairedInputParser(alg)
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
        nodes1 = self.alg.nodes1
        nodes2 = self.alg.nodes2
        nb_epoch = self.alg.nb_epoch
        batch_size = self.alg.batch_size
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
        numcount = 10
        for i in range(nb_epoch):
            # print epoch
            sys.stdout.write("Alg=%s, epoch=%d\r" % (self.alg, i))
            sys.stdout.flush()
            hist = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0, validation_data=(x, y))
            numcount-=1
            if (numcount!=0): continue
            numcount=10
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
        self.ip = LanguageModelInputParser()


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
            c_hatEnsemble  = self.chord2signatureOnehot((predOnehot+ self.chroma2WeightedOnehot(predChroma))/2.0)
            c_hatOnehotAvg = self.chord2signatureOnehot(predOnehotAvg)
            c_hatChromaAvg = self.chord2signatureChroma(predChromaAvg)
            c_hatEnsembleAvg = self.chord2signatureOnehot((predOnehotAvg+ self.chroma2WeightedOnehot(predChromaAvg))/2.0)
            errCntAvgOnehot    = np.average(np.abs(yChroma - c_hatOnehot)) * 12
            errCntAvgChroma    = np.average(np.abs(yChroma - c_hatChroma)) * 12
            errCntAvgEnsemble  = np.average(np.abs(yChroma - c_hatEnsemble)) * 12
            errCntAvgOnehotAvg = np.average(np.abs(yChroma - c_hatOnehotAvg)) * 12
            errCntAvgChromaAvg = np.average(np.abs(yChroma - c_hatChromaAvg)) * 12
            errCntAvgEnsembleAvg  = np.average(np.abs(yChroma - c_hatEnsembleAvg)) * 12
            np.save('../pred/' + filename + 'Onehot.npy', c_hatOnehot.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'Chroma.npy', c_hatChroma.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'Ensemble.npy', c_hatEnsemble.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'OnehotAvg.npy', c_hatOnehotAvg.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'ChromaAvg.npy', c_hatChromaAvg.astype(int).reshape((nb_test, seq_len, 12)))
            np.save('../pred/' + filename + 'EnsembleAvg.npy', c_hatEnsembleAvg.astype(int).reshape((nb_test, seq_len, 12)))
            errCntAvg = [errCntAvgOnehot, errCntAvgChroma, errCntAvgEnsemble, errCntAvgOnehotAvg, \
                         errCntAvgChromaAvg, errCntAvgEnsembleAvg]
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
