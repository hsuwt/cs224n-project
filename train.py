from util import *
from model import *
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(dest='algorithm', metavar='algorithm', nargs='?', default='BGRU_pair_L1_rand')
    parser.add_argument(dest='nodes1', nargs='?', type=int, default=64)
    parser.add_argument(dest='nodes2', nargs='?', type=int, default=64)
    parser.add_argument(dest='nb_epoch', nargs='?', type=int, default=200)
    parser.add_argument(dest='nb_epoch_pred', nargs='?', type=int, default=50)
    parser.add_argument(dest='dropout_rate', nargs='?', type=float, default=0.5)
    parser.add_argument(dest='batch_size', nargs='?', type=int, default=212)
    parser.add_argument(dest='nb_test', nargs='?', type=int, default=65)
    args = parser.parse_args()
    globals().update(vars(args))

    alg = parse_algorithm(args.algorithm)
    if 'LM' in alg:
        chord2signature = onehot2notes_translator() if 'one-hot' in alg else top3notes

    # M = training melody
    # m = testing melody
    # C = training chord progression
    # c = testing chord progression
    M, m, C, c = load_data(nb_test)
    x, y = get_XY(alg, m, c) # NOTE: after this the alg will have "one-hot-dim"

    nb_train = M.shape[0]
    model = build_model(alg, nodes1, nodes2, dropout_rate)
    history = [['epoch'], ['loss'],['val_loss'],['acc'],['val_acc']]
    # history will record the loss & acc of every epoch
    # since it's too time-consuming to compute the unique_idx and norms, record and save models after nb_epoch_pred epochs

    # X = training features
    # x = validation features (to evaluate val_loss & val_acc)
    # Y = training ground truth
    # y = validation ground truth
    # x_test = testing features (to evaluate unique_idx & norms)
    x_test = get_test(alg, m, C)
    
    es = EarlyStopping(monitor='val_loss', patience=2)
    for i in range(nb_epoch/nb_epoch_pred):
        for j in range(nb_epoch_pred):
            epoch = nb_epoch_pred*i+j+1
            sys.stdout.write("Alg=%s, epoch=%d\r" %(alg, epoch))
            sys.stdout.flush()
            X, Y = get_XY(alg, M, C)
            hist = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0, validation_data=(x, y))
            #hist = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0, validation_data=(x, y), callbacks=[es])

        # write history
        # history = write_history(history, hist, nb_epoch_pred * (i+1))
        # with open('history/' + alg + '_' + str(nodes1) + '_' + str(nodes2) + '.csv', 'w') as csvfile:
            # csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history)))

        # testing
        pred = np.array(model.predict(x_test))
        xdim = alg.get('one-hot-dim', 12)
        pred = pred.reshape((nb_test, 128, xdim))
        if 'LM' in alg:
            notes = chord2signature(pred)
            errCntAvg = np.average(np.abs(y - notes)) * 12
            with open('pred_LM.csv', 'w') as f:
                np.savetxt(f, notes.reshape((nb_test*128, 12)), delimiter=',', fmt="%d")
            print(errCntAvg)

        elif 'pair' in alg:
            pred = pred.reshape((nb_test, nb_train, 128))
            idx = np.argmax(np.sum(pred, 2), axis=1)
            c_hat = C[idx]
            bestN, uniqIdx, norm = print_result(c_hat, c, C, alg, False, 1)
            errCntAvg = np.average(np.abs(c_hat - c)) * 12
            with open('pred_pair.csv', 'w') as f:
                np.savetxt(f, c_hat.astype(int).reshape((nb_test*128, 12)), delimiter=',')
            print(errCntAvg)

            trn_loss = history[1][-1]
            val_loss = history[2][-1]
            trn_acc  = history[3][-1]
            val_acc  = history[4][-1]
            print("trn_loss=%.3f, trn_acc=%.3f" %(trn_loss, trn_acc))
            print("val_loss=%.3f, val_acc=%.3f" %(val_loss, val_acc))

            # record & save model
            record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
            #save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))

