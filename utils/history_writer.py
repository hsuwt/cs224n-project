import csv
import os


# history will record the loss of every epoch
# since it's too time-consuming to compute the unique_idx and norms,
# record and save models after nb_epoch_pred epochs)
class HistoryWriterPair(object):
    def __init__(self, csv_file):
        cols = ('epoch', 'loss', 'val_loss', 'errCntAvg', 'uniq_idx', 'norm')
        self.state = {k: list() for k in cols}
        self.new_state = {k: None for k in cols}
        self.writer = csv.DictWriter(csv_file, fieldnames=cols, lineterminator=os.linesep)

    def write_history(self, hist, epoch, err_count_avg, uniq_idx, norm):
        state = self.new_state
        state['epoch'] = epoch
        state['loss'] = round(hist.history['loss'][0], 2)
        state['val_loss'] = round(hist.history['val_loss'][0], 2)
        state['errCntAvg'] = round(err_count_avg, 2)
        state['uniq_idx'] = uniq_idx
        state['norm'] = norm
        self.writer.writerow(state)
        for k, v in state.items():
            self.state[k].append(v)

    def log(self):
        print "epoch: {epoch}, train_loss: {loss}, " \
              "test_loss: {val_loss}, errCntAvg: {errCntAvg}".format(**self.new_state)


class HistoryWriterLM(object):
    def __init__(self, csv_file):
        cols = ('epoch', 'train1', 'train12', 'val1', 'val12', 'err1', 'err12',
                'errEn', 'err1Avg', 'err12Avg', 'errEnAvg')
        self.state = {k: list() for k in cols}
        self.new_state = {k: None for k in cols}
        self.writer = csv.DictWriter(csv_file, fieldnames=cols, lineterminator=os.linesep)

    def write_history(self, hist, epoch, err):
        state = self.new_state
        state['epoch'] = epoch
        state['train1'] = round(hist.history['one-hot_loss'][0], 2)
        state['train12'] = round(hist.history['chroma_loss'][0], 2)
        state['val1'] = round(hist.history['val_one-hot_loss'][0], 2)
        state['val12'] = round(hist.history['val_chroma_loss'][0], 2)
        state['err1'] = round(err['err_count_avg'].onehot, 2)
        state['err12'] = round(err['err_count_avg'].chroma, 2)
        state['errEn'] = round(err['err_count_avg'].ensemble, 2)
        state['err1Avg'] = round(err['err_count_avg_avg'].onehot, 2)
        state['err12Avg'] = round(err['err_count_avg_avg'].chroma, 2)
        state['errEnAvg'] = round(err['err_count_avg_avg'].ensemble, 2)
        self.writer.writerow(state)
        for k, v in state.items():
            self.state[k].append(v)

    def log(self):
        print "epoch: {epoch}, train1: {train1}, train12: {train12}, " \
              "val1: {val1}, val12: {val12}. " \
              "err1: {err1}, err12: {err12}, err1Avg: {err1Avg}, err12Avg: {err12Avg}".format(**self.new_state)


class OneOffHistoryWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.written = False

    def write(self, contents):
        if self.written:
            raise RuntimeError('The file has already been written to! Writing again will overwrite previous results')
        self.written = True
        with open(self.filename, 'w') as ofile:
            for k, v in contents.items():
                ofile.write("{}={}\n".format(k, repr(v)))
