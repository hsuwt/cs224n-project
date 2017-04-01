"""
Build two reciprocal mappings between chord IDs and chord notes

Generate 2 files:
- chord-1hot-signatures.pickle:
  a dict that, given the bit-packed integer value of a chord notes vector, outputs a unique ID
- chord-1hot-signatures-rev.npy
  a numpy array with 12 columns, whose each row is a (unique) chord notes vector
The two mapping cancel each other

"""
import numpy as np
import pickle as pkl


class ChordNotes2OneHotTranscoder(object):
    """
    assume input C is in either 1 or 0
    """
    def __init__(self):
        with open('csv/chord-1hot-signatures.pickle') as pfile:
            self.sign2chord = pkl.load(pfile)
            self.size = len(self.sign2chord)

    def transcode(self, chord):
        N = chord.shape[0] * chord.shape[1]
        C2 = chord.reshape([N, 12]).astype(np.int)
        newC = np.zeros([N, self.size]).astype(np.int)
        indexes = np.packbits(C2, axis=1).astype(np.int)
        # packbits assumes numbers are in 8 bits. Instead our data uses 12 bits
        # therefore it is necessary to do the bit operaation below:
        indexes = (indexes[:, 0] << 4) + (indexes[:, 1] >> 4)
        chord_indexes = [self.sign2chord[i] for i in indexes]
        newC[np.arange(N), chord_indexes] = 1
        newC = newC.reshape([chord.shape[0], chord.shape[1], self.size])
        return newC


def chroma2Onehot(pred):
    chordId2sign = np.load('csv/chord-1hot-signatures-rev.npy')
    chordId2sign = chordId2sign / np.reshape(np.sum(chordId2sign, axis=1), (119,1))
    chordId2sign = np.nan_to_num(chordId2sign)
    def f(pred):
        pred = np.dot(pred, chordId2sign.T)

        maxes = np.amax(pred, axis=2)
        maxes = maxes.reshape(pred.shape[0], pred.shape[1], 1)

        e = np.exp(pred - maxes)
        sm = e / (np.sum(e, axis=2).reshape(pred.shape[0], pred.shape[1], 1))
        return np.nan_to_num(sm)
    return f


def get_onehot2chordnotes_transcoder():
    """
    generate a translator function that will map from a 1-hot repr of chord to a classical chord signature
    :return: f: the translator function
    """
    chordId2sign = np.load('csv/chord-1hot-signatures-rev.npy')

    def f(chord):
        """
        Translate from 1-hot array of dim {DIM} back to superimposed repr of dim=12
        :param chord: 1-hot representation of chords in (M, T, Dim)
        :return: chord signature in (M, T, 12)
        """
        M, T, Dim = chord.shape
        C2 = chord.reshape([M*T, Dim])
        index = np.argmax(C2, axis=1)
        newC = chordId2sign[index, :]
        return newC.reshape(M, T, 12)
    return f

def get_onehot2weighted_chords_transcoder():
    """
    generate a translator function that will map from a 1-hot repr of chord to a classical chord signature
    :return: f: the translator function
    """
    chordId2sign = np.load('csv/chord-1hot-signatures-rev.npy')

    def f(chord):
        """
        Translate from 1-hot array of dim {DIM} back to superimposed repr of dim=12
        :param chord: 1-hot representation of chords in (M, T, Dim)
        :return: chord signature in (M, T, 12)
        """
        M, T, Dim = chord.shape
        C2 = chord.reshape([M*T, Dim])
        weightedChords = np.dot(C2, chordId2sign)
        return weightedChords.reshape(M, T, 12)
    return f


def _load_data():
    """
    :return: chord data
    """
    C = np.genfromtxt('../csv/chord.csv', delimiter=',')  # Mx(T*12)
    C = np.swapaxes(C.reshape((C.shape[0],12,128)), 1, 2)  # MxTx12
    C = C.reshape((C.shape[0]*128, 12))  # (MT)x12
    return C


def _get_uniq(c):
    bytes = np.packbits(c.astype(np.int), axis=1).astype(np.int64)
    integer = (bytes[:, 0] << 4) + (bytes[:, 1] >> 4)
    uniqs = {integer: chord for integer, chord in zip(integer, c)}
    return uniqs


def _build_repr(uniqs):
    repr_set = {}
    reverse_set = []

    for i, (cid, chord) in enumerate(uniqs.items()):
        repr_set[cid] = i
        reverse_set.append(chord)
    return repr_set, np.array(reverse_set)


def _savemap(amap, path):
    with open(path, 'wb') as pfile:
        pkl.dump(amap, pfile)


def _saveunmap(unmap, path):
    np.save(path, unmap)


if __name__ == '__main__':
    c = _load_data()
    cs = _get_uniq(c)
    print "Discovered %d different chord signatures, out of %d possibilities" % (len(cs), 2 << 12)
    amap, unmap = _build_repr(cs)
    _savemap(amap, 'csv/chord-1hot-signatures.pickle')
    _saveunmap(unmap, 'csv/chord-1hot-signatures-rev')
