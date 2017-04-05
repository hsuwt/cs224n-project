# from util import top3notes, InputParser
from util import *
from utils.load_data import *
from utils.build_chord_repr import *
from utils.history_writer import *
from tqdm import trange
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt


class TrainingStrategy(object):
    def __init__(self):
        pass

    def train(self, model):
        raise NotImplementedError('override this!')

    def test(self, model):
        raise NotImplementedError('override this!')

    @staticmethod
    def get_filename(_alg):
        major = _alg.strategy
        minor = 'onehot' if 'one-hot' in _alg.model else ''
        rnn = 'RNN' if 'RNN' in _alg.model else "GRU" if "GRU" in _alg.model else "LSTM" if "LSTM" in _alg.model else ''
        if 'Bidirectional' in _alg.model:
            rnn += 'B'

        fn = rnn + '_' + major
        if minor:
            fn += '_' + minor
        if 'sample-biased' in _alg.model:
            fn += '_' + 'sb'
        fn += '_nodes' + str(_alg.nodes1)
        if 'mtl_ratio' in _alg:
            fn += '_' + str(_alg.mtl_ratio)
        return fn


class LanguageModelTrainingStrategy(TrainingStrategy):
    def __init__(self, args):
        super(LanguageModelTrainingStrategy, self).__init__()
        self.args = args
        args = self.args
        self.chord2signatureOnehot = get_onehot2chordnotes_transcoder()
        self.chroma2WeightedOnehot = get_onehot2weighted_chords_transcoder()
        self.chord2signatureChroma = top3notes

        # Naming Guide:
        # M = training melody
        # m = testing melody
        # C = training chord progression
        # c = testing chord progression
        self.ip = LanguageModelInputParser()

        # Naming Guide
        # X = training features
        # x = validation features (to evaluate val_loss & val_acc)
        # Y = training ground truth
        # y = validation ground truth
        # x_test = testing features (to evaluate unique_idx & norms)
        nb_test = args.nb_test
        data = load_data(args, nb_test)
        train_data = data['train']
        test_data = data['test']

        DataSet = namedtuple('DataSet', ['x', 'y_chroma', 'y_onehot', 'sw'])
        x, y12, y1 = self.ip.get_XY(train_data.melody, train_data.chord)
        self.trainset = DataSet(x=x, y_chroma=y12, y_onehot=y1, sw=train_data.sw)
        x, y12, y1 = self.ip.get_XY(test_data.melody, test_data.chord)
        self.testset = DataSet(x=x, y_chroma=y12, y_onehot=y1, sw=test_data.sw)

        self.x_test = test_data.melody
        self.seq_len = 128
        self.ydim = self.trainset.y_onehot.shape[2]
        self.nb_train = train_data.melody.shape[0]
        self.nb_test = test_data.melody.shape[0]

    def get_ydim(self):
        return self.ydim

    def train(self, model):
        if self.args.debug:
            nb_epoch = self.args.nb_epoch
        else:
            nb_epoch = 1
        batch_size = self.args.batch_size
        seq_len = self.seq_len
        nb_test = self.nb_test
        train = self.trainset
        test = self.testset
        x_test = self.x_test
        filename = self.get_filename(self.args)

        Pair = namedtuple('Pair', ['onehot', 'chroma'])
        Triple = namedtuple('Triple', ['onehot', 'chroma', 'ensemble'])

        def apply_triple(f, triple):
            return Triple(onehot=f(triple.onehot), chroma=f(triple.chroma), ensemble=f(triple.ensemble))

        with open('history/' + filename + '.csv', 'w') as csvfile:
            history = HistoryWriterLM(csvfile)
            pbar = trange(nb_epoch)
            print test.x.shape
            for i in pbar:
                hist = model.fit(train.x, {'one-hot': train.y_onehot, 'chroma': train.y_chroma},
                                 nb_epoch=1, verbose=0,
                                 batch_size=batch_size,
                                 sample_weight={'one-hot': train.sw, 'chroma': train.sw},
                                 validation_data=(test.x,
                                                  {'one-hot': test.y_onehot, 'chroma': test.y_chroma},
                                                  {'one-hot': test.sw, 'chroma': test.sw}))
                # testing
                pred = Pair(*model.predict(x_test))
                pred_avg = Pair(onehot=smooth(np.array(pred.onehot)),
                                chroma=smooth(np.array(pred.chroma)))
            
                # signature here refers to theo output feature vector to be used for training
                c_hat = Triple(onehot=self.chord2signatureOnehot(pred.onehot),
                               chroma=self.chord2signatureChroma(pred.chroma),
                               ensemble=self.chord2signatureChroma(
                                   (pred.chroma + self.chroma2WeightedOnehot(pred.onehot)) / 2.))
                c_hat_avg = Triple(onehot=self.chord2signatureOnehot(pred_avg.onehot),
                                   chroma=self.chord2signatureChroma(pred_avg.chroma),
                                   ensemble=self.chord2signatureChroma(
                                       (pred_avg.chroma + self.chroma2WeightedOnehot(pred_avg.onehot)) / 2.))
                
                err_count_avg = apply_triple(lambda chat: np.average(np.abs(test.y_chroma - chat)) * 12, c_hat)
                err_count_avg_avg = apply_triple(lambda chat_avg: np.average(np.abs(test.y_chroma - chat_avg)) * 12,
                                                 c_hat_avg)

                base = '../pred/' + filename + '{}.npy'
                for _c, fn in zip(c_hat, ('Onehot', 'Chroma', 'Ensemble')):
                    np.save(base.format(fn), _c.astype(int).reshape((nb_test, seq_len, 12)))
                for cavg, fn in zip(c_hat_avg, ('OnehotAvg', 'ChromaAvg', 'EnsembleAvg')):
                    np.save(base.format(fn), cavg.astype(int).reshape((nb_test, seq_len, 12)))

                history.write_history(hist, i+1, {'err_count_avg': err_count_avg, 'err_count_avg_avg': err_count_avg_avg})
                ns = history.new_state
                pbar.set_postfix(train_loss=(ns['train1'], ns['train12']),
                                 test_loss=(ns['val1'], ns['val12']),
                                 errCntAvg=(ns['err1'], ns['err12']))
                
                # error analysis
                pred_oh_erranaly = np.argmax(pred.onehot, axis=2).flatten() # (30 x 128,)
                test_y_oh_erranaly = np.argmax(test.y_onehot,axis=2).flatten() # (30 x 128,)
                print classification_report(test_y_oh_erranaly, pred_oh_erranaly)
                print confusion_matrix(test_y_oh_erranaly,pred_oh_erranaly)                    
                


class IterativeImproveStrategy(TrainingStrategy):
    def __init__(self, args):
        super(IterativeImproveStrategy, self).__init__()
        self.args = args
        self.ip = PairedInputParser(args)
        # Naming Guide
        # X = training features
        # x = validation features (to evaluate val_loss & val_acc)
        # Y = training ground truth
        # y = validation ground truth
        # x_test = testing features (to evaluate unique_idx & norms)
        self.nb_test = args.nb_test
        self.c2o_transcoder = ChordNotes2OneHotTranscoder()
        data = load_data(args, self.nb_test)
        train_data = data['train']
        test_data = data['test']

        DataSet = namedtuple('DataSet', ['x', 'y', 'sw'])
        x, y = self.ip.get_XY(train_data.melody, train_data.chord)
        self.trainset = DataSet(x, y, np.tile(train_data.sw, (2, 1)))
        x, y = self.ip.get_XY(test_data.melody, test_data.chord)
        self.testset = DataSet(x, y, np.tile(test_data.sw, (2, 1)))
        self.x_test = get_test(args, m=test_data.melody, M=train_data.melody, C=train_data.chord)
        self.test_chord, self.train_chord = test_data.chord, train_data.chord
        self.test_melody, self.test_melody = test_data.melody, train_data.melody
        self.seq_len = 128
        self.nb_train = train_data.melody.shape[0]
        self.nb_test = test_data.melody.shape[0]
        self.test_freq = 20
        self.num_iter = 0

        # KNN baseline
        self.best_matches = select_closest_n(m=test_data.melody, M=train_data.melody)
        self.best1 = self.best_matches[:, 0].ravel()
        self.knn_err_count_avg = np.average(np.abs(train_data.chord[self.best1] - test_data.chord)) * 12

    def train(self, model):
        if self.args.debug:
            nb_epoch = 1
            batch_size = self.args.batch_size
            test_freq = 1
        else:
            nb_epoch = self.args.nb_epoch
            batch_size = self.args.batch_size
            test_freq = self.test_freq

        args = self.args
        train = self.trainset
        test = self.testset
        x_test = self.x_test
        test_chord, train_chord = self.test_chord, self.train_chord
        test_melody = self.test_melody
        nb_train, nb_test = self.nb_train, self.nb_test
        
        # generate frequency histogram
        test_chord_oh = self.c2o_transcoder.transcode(test_chord)
        test_chord_oh = np.argmax(test_chord_oh, axis=2)    
        hg_labels, hg_values = zip(*Counter(test_chord_oh.flatten()).items())
        hg_indexes = np.arange(len(hg_labels))
        width = 0.5
        plt.bar(hg_indexes, hg_values, width)
        plt.xticks(hg_indexes + width * 0.5-0.25, hg_labels, fontsize=7) 
        plt.savefig("freq_histogram.png")

        filename = self.get_filename(self.args)
        print "Result will be written to history/{0}.csv and {0}-results.txt".format(filename)
        with open('history/' + filename + '.csv', 'w') as csvfile:
            history = HistoryWriterPair(csvfile)
            pbar = trange(nb_epoch)
            pbar.set_postfix(train_loss='_', test_loss='_', errCntAvg='_')
            for i in pbar:
                hist = model.fit(train.x, train.y,
                                 nb_epoch=1, verbose=0,
                                 batch_size=batch_size,
                                 validation_data=(test.x, test.y))
                if i % self.test_freq == 0: #test_freq - 1:
                    pred = np.array(model.predict(x_test)).reshape((nb_test, -1, 128, 24))
                    errs = np.sum(pred, axis=(2,3))
                    idx = np.argmin(errs, axis=1)  # 100,
                    c_hat = train_chord[idx].astype(int)  # 100, 128, 12
                    if 'correct' in args.model:
                        corrected = c_hat + 0
                        pred = pred[np.arange(nb_test), idx]
                        # iteratively improve corrected
                        for j in range(self.num_iter):
                            pred_add = pred[:,:,:12]
                            pred_delete = pred[:,:,-12:]
                            pred_add *= (1-corrected)
                            pred_delete *= corrected
                            pred_add, pred_delete = smooth(pred_add), smooth(pred_delete)
                            max_add, max_delete = np.amax(pred_add), np.amax(pred_delete)
                            corrected[pred_add==max_add] = 1
                            corrected[pred_delete==max_delete] = 0
                            np.save('../pred/' + filename + 'Corrected' + str(j) +'.npy', corrected)
                            np.save('../pred/' + filename + 'CorrectedAvg' + str(j) + '.npy', smooth(corrected))
                            x_test_correct = np.concatenate((test_melody[idx], corrected), 2)
                            pred = np.array(model.predict(x_test_correct)).reshape((nb_test, 128, 24))
                            # err_count_avg = np.average(np.abs(corrected - test_chord)) * 12
                    bestN, uniq_idx, norm = print_result(c_hat, test_chord, train_chord, args, 1)
                    err_count_avg = np.average(np.abs(c_hat - test_chord)) * 12
                    np.save('../pred/' + filename + '.npy', c_hat)

                    history.write_history(hist, i+1, err_count_avg, uniq_idx, norm)
                    pbar.set_postfix(train_loss=history.new_state['loss'],
                                     test_loss=history.new_state['val_loss'],
                                     errCntAvg=history.new_state['errCntAvg'])
                    
                    # error analysis
                    c_hat_oh = self.c2o_transcoder.transcode(c_hat)
                    c_hat_oh = np.argmax(c_hat_oh, axis=2)
            
                    print (c_hat_oh== test_chord_oh).sum()
                    print c_hat_oh
                    print test_chord_oh
                    print classification_report(c_hat_oh.flatten(), test_chord_oh.flatten())
                    print confusion_matrix(c_hat_oh.flatten(), test_chord_oh.flatten()).shape                    
                    
                    

        knn_precision = (idx == self.best1).sum() / float(nb_test)
        knn_recall = sum(1 for i in range(nb_test) if idx[i] in self.best_matches[i]) / float(nb_test)
        results = OneOffHistoryWriter('history/' + filename + '-results.txt')
        results.write({'knn_errCntAvg (How much KNN best matches defer from real ground truth)': self.knn_err_count_avg,
                       'Precision (Portion of KNN best matches that agree with NN result)': knn_precision,
                       'Recall (How many NN results are included in the 100 songs selected by KNN)': knn_recall,
                       'idx': idx})
