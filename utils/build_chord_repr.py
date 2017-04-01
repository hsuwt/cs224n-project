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
        n = chord.shape[0] * chord.shape[1]
        chord2 = chord.reshape([n, 12]).astype(np.int)
        new_chord = np.zeros([n, self.size]).astype(np.int)
        indexes = np.packbits(chord2, axis=1).astype(np.int)
        # packbits assumes numbers are in 8 bits. Instead our data uses 12 bits
        # therefore it is necessary to do the bit operaation below:
        indexes = (indexes[:, 0] << 4) + (indexes[:, 1] >> 4)
        chord_indexes = [self.sign2chord[i] for i in indexes]
        new_chord[np.arange(n), chord_indexes] = 1
        new_chord = new_chord.reshape([chord.shape[0], chord.shape[1], self.size])
        return new_chord


# TODO: what is this? How is this different from ChordNotes2OneHotTranscoder
def chroma2onehot():
    chord_id2sign = np.load('csv/chord-1hot-signatures-rev.npy')
    chord_id2sign = chord_id2sign / np.reshape(np.sum(chord_id2sign, axis=1), (119, 1))
    chord_id2sign = np.nan_to_num(chord_id2sign)

    def f(pred):
        pred = np.dot(pred, chord_id2sign.T)

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
    chord_id2sign = np.load('csv/chord-1hot-signatures-rev.npy')

    def f(chord):
        """
        Translate from 1-hot array of dim {DIM} back to superimposed repr of dim=12
        :param chord: 1-hot representation of chords in (M, T, Dim)
        :return: chord signature in (M, T, 12)
        """
        m, length, dim = chord.shape
        chord2 = chord.reshape([m*length, dim])
        index = np.argmax(chord2, axis=1)
        new_chord = chord_id2sign[index, :]
        return new_chord.reshape(m, length, 12)
    return f


def get_onehot2weighted_chords_transcoder():
    """
    generate a translator function that will map from a 1-hot repr of chord to a classical chord signature
    :return: f: the translator function
    """
    chord_id2sign = np.load('csv/chord-1hot-signatures-rev.npy')

    def f(chord):
        """
        Translate from 1-hot array of dim {DIM} back to superimposed repr of dim=12
        :param chord: 1-hot representation of chords in (M, T, Dim)
        :return: chord signature in (M, T, 12)
        """
        m, length, dim = chord.shape
        chord2 = chord.reshape([m*length, dim])
        weighted_chords = np.dot(chord2, chord_id2sign)
        return weighted_chords.reshape(m, length, 12)
    return f


def _load_data():
    """
    :return: chord data
    """
    c = np.genfromtxt('../csv/chord.csv', delimiter=',')  # Mx(T*12)
    c = np.swapaxes(c.reshape((c.shape[0], 12, 128)), 1, 2)  # MxTx12
    c = c.reshape((c.shape[0]*128, 12))  # (MT)x12
    return c


def _get_uniq(c):
    _bytes = np.packbits(c.astype(np.int), axis=1).astype(np.int64)
    integer = (_bytes[:, 0] << 4) + (_bytes[:, 1] >> 4)
    uniqs = {integer: chord for integer, chord in zip(integer, c)}
    return uniqs


def _build_repr(uniqs):
    repr_set = {}
    reverse_set = []

    for i, (cid, chord) in enumerate(uniqs.items()):
        repr_set[cid] = i
        reverse_set.append(chord)
    return repr_set, np.array(reverse_set)


def _savemap(_amap, path):
    with open(path, 'wb') as pfile:
        pkl.dump(_amap, pfile)


def _saveunmap(_umap, path):
    np.save(path, _umap)


if __name__ == '__main__':
    chords = _get_uniq(_load_data())
    print "Discovered %d different chord signatures, out of %d possibilities" % (len(chords), 2 << 12)
    amap, unmap = _build_repr(chords)
    _savemap(amap, 'csv/chord-1hot-signatures.pickle')
    _saveunmap(unmap, 'csv/chord-1hot-signatures-rev')
