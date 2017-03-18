from util import *
from model import *

if __name__ == "__main__":
    if len(sys.argv) < 2: alg = 'GRU_root6'
    else: alg = sys.argv[1]
    if len(sys.argv) < 3: batch_size = 212
    else: batch_size = sys.argv[2]
    if len(sys.argv) < 4: nb_test = 100
    else: nb_test = sys.argv[3]

    # M = training melody
    # m = testing melody
    # C = training chord progression
    # c = testing chord progression
    M, m, C, c = load_data(nb_test)
    nb_train = M.shape[0]
    model = load_model(alg)
    ip = InputParser(alg)
    x, y = ip.get_XY(m, c)
    x_te = get_test(alg, m, C)

    # make prediction
    idx = np.argmax(np.array(model.predict(x_te))[:,0].reshape((nb_test, nb_train)), axis=1)
    pred   = C[idx]
    bestN, uniqIdx, norm = print_result(pred, c, C, alg, True, 1)
    val_loss, val_acc = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    print("val_loss=%.3f, val_acc=%.3f" %(val_loss, val_acc))

