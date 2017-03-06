from util import *
from model import *

def pred(melody, alg):
    # melody.shape = (128, 12)
    n_test = 100
    M, m, C, c = load_data(n_test)
    m = np.reshape(melody, (1, 128, 12))
    n_train = M.shape[0]
    model = load_model(alg)
    x, y = get_XY(alg, m, c)
    x_te = get_test(alg, m, C)
    idx = np.argmax(np.array(model.predict(x_te))[:,0].reshape((n_test, n_train)), axis=1)
    pred = C[idx]
    bestN, uniqIdx, norm = print_result(pred, c, C, alg, 0, 5)
    return bestN[0]
