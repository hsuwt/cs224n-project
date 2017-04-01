from build_chord_repr import ChordNotes2OneHotTranscoder, get_onehot2chordnotes_transcoder, chroma2Onehot
from collections import namedtuple
import numpy as np
import csv
import os


Data = namedtuple('Data', ['melody', 'chord', 'sw'])

def load_data(alg, nb_test):
    chord, melody, sw = parse_data(alg, 128)
    return {'train': Data(melody=melody[:-nb_test], chord=chord[:-nb_test], sw=sw[:-nb_test]),
            'test': Data(melody=melody[-nb_test:], chord=chord[-nb_test:], sw=sw[-nb_test:])}


class InputParser(object):
    """
    This replaces previous function GetXY
    """

    def __init__(self):
        raise NotImplementedError('inherit me')

    def get_XY(self, M, C):
        raise NotImplementedError('override me!')


class LanguageModelInputParser(InputParser):
    """
    This replaces previous function GetXY
    """

    def __init__(self):
        self.transcoder = ChordNotes2OneHotTranscoder()

    def get_XY(self, M, C):
        COnehot = self.transcoder.transcode(C)
        return M, C, COnehot


class PairedInputParser(InputParser):
    """
    This replaces previous function GetXY
    """

    def __init__(self, alg):
        self.alg = alg
        self.transcoder = ChordNotes2OneHotTranscoder()

    def get_XY(self, M, C):
        COnehot = self.transcoder.transcode(C)
        n = M.shape[0]
        idx = np.random.randint(n, size=n)
        C_neg, C_negOnehot = C[idx], COnehot[idx]
        Ones = np.ones((n, 128, 1))
        Zeros = np.zeros((n, 128, 1))
        assert 'L1diff' in self.alg.model

        L1diff, L1diffOnehot = (C_neg - C) / 2. + 0.5, (C_negOnehot - COnehot) / 2. + 0.5
        if 'rand' in self.alg.model:
            X = np.concatenate((M, C_neg), 2)
            Y = L1diff
        else:
            MC_neg = np.concatenate((M, C_neg), 2)
            MC = np.concatenate((M, C), 2)
            X = np.concatenate((MC, MC_neg), 0)
            Y = np.concatenate((np.tile(Zeros, 12) + 0.5, L1diff), 0)
            return X, Y
        Y = 1 - Y / 12.0
        return X, Y


def get_test(strategy, m, C):
    # x_te are the final testing features to match m to C
    if strategy == 'pair' or strategy == 'correct':
        m_rep, C_rep = rep(m, C)
        return np.concatenate((m_rep, C_rep), 2)
    elif strategy == 'LM':
        return m
    else:
        raise ValueError('Invalid strategy %s' % strategy)

def rep(m, C):
    nb_test = m.shape[0]
    nb_train = C.shape[0]
    C_rep = np.tile(C, (nb_test, 1, 1))
    m_rep = np.tile(m, (nb_train, 1, 1))
    m_rep = np.reshape(m_rep, (nb_train, nb_test, 128, 12))
    m_rep = np.swapaxes(m_rep, 1, 0)
    m_rep = np.reshape(m_rep, (nb_test * nb_train, 128, 12))
    return m_rep, C_rep

def parse_data(alg, max_length):
    with open('csv/npy-exists.config') as keyvaluefile:
        avaialable = [key[0] for key in csv.reader(keyvaluefile)]
        if 'sample-biased' not in alg.model and 'normal' in avaialable:
            print "I can load normal set of params from npy files"
            M = np.load('csv/normal-melody.npy')
            C = np.load('csv/normal-chord.npy')
            SW = np.load('csv/normal-sampleweight.npy')
            return C, M, SW

        if 'sample-biased' in alg.model and 'sample-biased' in avaialable:
            print "I can load sample-biased set of params from npy files"
            M = np.load('csv/sample-biased-melody.npy')
            C = np.load('csv/sample-biased-chord.npy')
            SW = np.load('csv/sample-biased-sampleweight.npy')
            return C, M, SW

    C = np.genfromtxt('csv/chord.csv', delimiter=',')
    # Data in melody.csv and root.csv are represented as [0,11].
    # Thus, we first span it to boolean matrix
    M_dense = np.genfromtxt('csv/melody.csv', delimiter=',')
    assert (M_dense.shape[1] * 12 == C.shape[1])
    M = np.zeros((M_dense.shape[0], M_dense.shape[1] * 12))
    sample_weight = np.zeros(M_dense.shape)  # Use this if using variable length
    for i in range(M_dense.shape[0]):
        for j in range(M_dense.shape[1]):
            if not np.isnan(M_dense[i][j]):
                notes = int(M_dense[i][j])
                M[i][M_dense.shape[1] * notes + j] = 1
                sample_weight[i][j] = 1
    C = np.nan_to_num(C)
    M = np.swapaxes(M.reshape((M_dense.shape[0], 12, M_dense.shape[1])), 1, 2)
    C = np.swapaxes(C.reshape((C.shape[0], 12, -1)), 1, 2)

    sample_weight = np.ones((C.shape[0], C.shape[1]))
    if 'sample-biased' in alg.model:
        for p in range(1, 8):
            sample_weight[:, ::2 ** p] += 1
        sample_weight /= 8.0

    # store
    with open('csv/npy-exists.config', 'w') as keyvaluefile:
        avaialable = csv.writer(keyvaluefile)
        key = 'sample-biased' if 'sample-biased' in alg.model else 'normal'
        avaialable.writerow([key])
        np.save('csv/' + key + '-chord', C)
        np.save('csv/' + key + '-melody', M)
        np.save('csv/' + key + '-sampleweight', sample_weight)
    return C, M, sample_weight


def parse_big_data(alg, max_length):
    nb_train = sum([len(files) for r, d, files in os.walk("../dataset/melody")])
    C = np.zeros((nb_train, max_length, 12))
    M = np.zeros((nb_train, max_length, 12))
    sample_weight = np.zeros((nb_train, max_length))
    train_idx = 0
    for root, _, files in os.walk("../dataset/melody"):
        for m_file_name in files:
            m_file_name_path = os.path.join(root, m_file_name)
            c_file_name_path = m_file_name_path.replace("/melody/", "/chord/", 1)
            m_file_matrix = np.genfromtxt(m_file_name_path, delimiter=',')
            c_file_matrix = np.genfromtxt(c_file_name_path, delimiter=',')
            if len(m_file_matrix) == 0 or len(c_file_matrix) == 0: continue
            m_file_matrix = m_file_matrix[:, :max_length]
            c_file_matrix = c_file_matrix[:, :max_length]
            seq_len = min(m_file_matrix.shape[1], c_file_matrix.shape[1])
            tmp_m, tmp_c = np.zeros((12, seq_len)), np.zeros((12, seq_len))
            for i in range(128 / 12):
                tmp_m = np.logical_or(tmp_m, m_file_matrix[i * 12:(i + 1) * 12, :seq_len])
                tmp_c = np.logical_or(tmp_c, c_file_matrix[i * 12:(i + 1) * 12, :seq_len])
            sample_weight[train_idx, :seq_len] = np.ones((1, seq_len))
            M[train_idx, :seq_len], C[train_idx, :seq_len] = np.swapaxes(tmp_m, 0, 1), np.swapaxes(tmp_c, 0, 1)
            train_idx += 1
    C = C[:train_idx]
    M = M[:train_idx]
    sample_weight = sample_weight[:train_idx]
    return C, M, sample_weight