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
    cs = set(str(x) for x in c)
    return cs

def build_repr(s):
    N = len(cs)
    repr_set = {}
    reverse_set = []
    for i, x in enumerate(s):
        repr_set[x] = i
        reverse_set.append(x)
    return repr_set, reverse_set

if __name__ == '__main__':
    c = load_data()
    cs = get_uniq(c)
    print "Discovered %d different chord signatures, out of %d possibilities" % (len(cs), 2<<12)
    r, rev = build_repr(cs)
    with open('csv/chord-1hot-signatures.pickle', 'wb') as pfile:
        pkl.dump(r, pfile)
    with open('csv/chord-1hot-signatures-rev.pickle', 'wb') as pfile:
        pkl.dump(rev, pfile)
