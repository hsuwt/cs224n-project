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
    def __init__(self):
        with open('csv/chord-1hot-signatures.pickle') as pfile:
            self.sign2chord = pkl.load(pfile)
            self.size = len(self.sign2chord)

    def transcode(self, C):
        N = C.shape[0] * C.shape[1]
        C2 = C.reshape([N, 12]).astype(np.int)
        newC = np.zeros([N, self.size]).astype(np.int)
        indexes = np.packbits(C2, axis=1).astype(np.int)
        # packbits assumes numbers are in 8 bits. Instead our data uses 12 bits
        # therefore it is necessary to do the bit operaation below:
        indexes = (indexes[:, 0] << 4) + (indexes[:, 1] >> 4)
        chord_indexes = [self.sign2chord[i] for i in indexes]
        newC[np.arange(N), chord_indexes] = 1
        newC = newC.reshape([C.shape[0], C.shape[1], self.size])
        return newC

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
        C2 =  chord.reshape([M*T, Dim])
        index = np.argmax(C2, axis=1)
        newC = chordId2sign[index, :]
        return newC.reshape(M, T, 12)
    return f


def load_data():
    """

    :param alg: algorithm. Not used
    :param nb_test: size of test set, should be less than the total number of data entry
    :return:
    """
    C = np.genfromtxt('csv/chord.csv', delimiter=',') # Mx(T*12)
    C = np.swapaxes(C.reshape((C.shape[0],12,128)), 1, 2) # MxTx12
    C = C.reshape((C.shape[0]*128, 12)) # (MT)x12
    return C

def get_uniq(c):
    bytes = np.packbits(c.astype(np.int), axis=1).astype(np.int64)
    integer = (bytes[:, 0] << 4) + (bytes[:, 1] >> 4)
    uniqs = {integer: chord for integer, chord in zip(integer, c)}
    return uniqs

def build_repr(uniqs):
    repr_set = {}
    reverse_set = []

    for i, (cid, c) in enumerate(uniqs.items()):
        repr_set[cid] = i
        reverse_set.append(c)
    return repr_set, np.array(reverse_set)

def savemap(map, path):
    with open(path, 'wb') as pfile:
        pkl.dump(map, pfile)

def saveunmap(unmap, path):
    np.save(path, unmap)

if __name__ == '__main__':
    c = load_data()
    cs = get_uniq(c)
    print "Discovered %d different chord signatures, out of %d possibilities" % (len(cs), 2<<12)
    map, unmap = build_repr(cs)
    savemap(map, 'csv/chord-1hot-signatures.pickle')
    saveunmap(unmap, 'csv/chord-1hot-signatures-rev')
