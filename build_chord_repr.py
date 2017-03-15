import numpy as np
import pickle as pkl

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
