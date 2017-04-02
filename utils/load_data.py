from build_chord_repr import ChordNotes2OneHotTranscoder
from collections import namedtuple
import numpy as np
import csv
import os
from util import dataAug, rotateNotes

Data = namedtuple('Data', ['melody', 'chord', 'sw'])


def load_data(alg, nb_test):
    chord, melody, sw = parse_data(alg, 128)
    # Up sample
    chord_us, melody_us, sw_us = np.repeat(chord, 2, axis=1), np.repeat(melody, 2, axis=1), np.repeat(sw,2, axis=1)
    chord = np.concatenate((chord, chord_us[:,:128,:], chord_us[:,128:,:]), axis=0)
    melody = np.concatenate((melody, melody_us[:,:128,:], chord_us[:,128:,:]), axis=0)
    sw = np.concatenate((sw, sw_us[:,:128], sw_us[:,128:]), axis=0)
    
    tmp = dataAug(chord)
    return {'train': Data(melody=dataAug(melody[:-nb_test]), chord=dataAug(chord[:-nb_test]), sw=np.tile(sw[:-nb_test], (12,1))),
            'test': Data(melody=dataAug(melody[-nb_test:]), chord=dataAug(chord[-nb_test:]), sw=np.tile(sw[-nb_test:], (12,1)))}


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
        n = M.shape[0]
        idx = np.random.randint(n, size=n)
        C_neg = C[idx]
        Ones = np.ones((n, 128, 1))
        Zeros = np.zeros((n, 128, 1))
        add    = C - C_neg == 1
        delete = C_neg - C == 1

        MC_pos = np.concatenate((M, C), 2)
        MC_neg = np.concatenate((M, C_neg), 2)
        X = np.concatenate((MC_pos, MC_neg), 0)

        YAdd    = np.concatenate((np.tile(Zeros, 12), add), 0)
        YDelete = np.concatenate((np.tile(Zeros, 12), delete), 0)
        Y = np.concatenate((YAdd, YDelete), 2)
        return X, Y

def get_test(args, m, M, C):
    """
    Create a testing feature for training
    :param strategy:
    :param m: test melody     (nb_test,  128, 12)
    :param M: train melody    (nb_train, 128, 12)
    :param C: train melody    (nb_train, 128, 12)

    First select best nb_test matching songs, then filter corresponding chords
    Let the filtered chord be D (nb_test * nb_test, 128, 12)

    :return: m[] + D : (nb_test * nb_test, 128, 24)
    """
    if args.strategy == 'pair' and 'knn' in args.model:
        nb_test = m.shape[0]
        best_indexes = select_closest_n(m, M)
        chord_expanded = C[best_indexes.ravel()]
        melody_repeated = np.repeat(m, nb_test, axis=0)
        return np.concatenate((melody_repeated, chord_expanded), 2)  # (nb_test * nb_test, 128, 24)
    elif args.strategy == 'pair':
        m_rep, C_rep = rep(m, C)
        return np.concatenate((m_rep, C_rep), 2)
    elif args.strategy == 'LM':
        return m
    else:
        raise ValueError('Invalid strategy %s' % args.strategy)

def rep(m, C):
    nb_test = m.shape[0]
    nb_train = C.shape[0]
    C_rep = np.tile(C, (nb_test,  1, 1))
    m_rep = np.repeat(m, nb_train, axis=0)
    return m_rep, C_rep

def select_closest_n(m, M):
    """
    Given test melody and train melody, return {n} indexes of train melody that are closest
    to each test songs
    This does so by computing
    :param m: test melody     (nb_test,  128, 12)
    :param M: train melody    (nb_train, 128, 12)
    :return: indexes of {n} best matching training songs for each teset songs (nb_test, n)
    """
    nb_test = m.shape[0]
    best_match = np.array([(m[i] * M).sum(axis=(1, 2)).argsort()[::-1][:nb_test] for i in range(nb_test)], dtype=np.int)
    return best_match


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