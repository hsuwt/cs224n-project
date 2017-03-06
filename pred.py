from util import *
from model import *

def pred(melody, alg):
    # melody.shape = (128, 12)
    n_test = 100
    M, m, C, c, R, r = load_data(alg, n_test)
    m = np.reshape(melody, (1, 128, 12))
    n_train = M.shape[0]
    model = load_model(alg)
    x, y = get_XY(alg, m, c, r)
    x_te = get_test(alg, m, C, R)
    if 'baseline' in alg:
        pred = np.array(model.predict(x_te))
    else:
        idx = np.argmax(np.array(model.predict(x_te))[:,0].reshape((n_test, n_train)), axis=1)
        if 'weighted' in alg:
            C[C > 0] = 1
            c[c > 0] = 1
        pred = C[idx]
    bestN, uniqIdx, norm = print_result(pred, c, C, alg, 0, 5)
    return bestN[0]
