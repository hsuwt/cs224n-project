from util import *
from model import *
import time

if __name__ == "__main__":
    if len(sys.argv) < 2: alg = 'GRU_pair_L2_rand'
    else: alg = sys.argv[1]
    if len(sys.argv) < 3: nodes1 = 64
    else: nodes1 = int(sys.argv[2])
    if len(sys.argv) < 4: nodes2 = 64
    else: nodes2 = int(sys.argv[3])
    if len(sys.argv) < 5: nb_epoch = 400
    else: nb_epoch = int(sys.argv[4])
    if len(sys.argv) < 6: nb_epoch_pred = 20
    else: nb_epoch_pred = int(sys.argv[5])
    if len(sys.argv) < 7: dropout_rate = 0.5
    else: dropout_rate = float(sys.argv[6])
    if len(sys.argv) < 8: batch_size = 212
    else: batch_size = int(sys.argv[7])
    if len(sys.argv) < 9: nb_test = 65
    else: nb_test = int(sys.argv[8])

    # M = training melody
    # m = testing melody
    # C = training chord progression
    # c = testing chord progression
    M, m, C, c = load_data(alg, nb_test)
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
    x, y = get_XY(alg, m, c)
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
        history = write_history(history, hist, nb_epoch_pred * (i+1))
        # with open('history/' + alg + '_' + str(nodes1) + '_' + str(nodes2) + '.csv', 'w') as csvfile:
            # csv.writer(csvfile, lineterminator=os.linesep).writerows(map(list, zip(*history)))

        # testing
        if 'baseline' in alg:
            pred = np.array(model.predict(x_test))
        else:
            if '128' in alg:
                idx = np.argmax(np.sum(np.array(model.predict(x_test)).reshape((nb_test, nb_train, 128)), 2), axis=1)
            else:
                idx = np.argmax(np.array(model.predict(x_test)).reshape((nb_test, nb_train)), axis=1)
            pred = C[idx]
        bestN, uniqIdx, norm = print_result(pred, c, C, alg, True, 1)

        trn_loss = history[1][-1]
        val_loss = history[2][-1]
        trn_acc  = history[3][-1]
        val_acc  = history[4][-1]
        print("trn_loss=%.3f, trn_acc=%.3f" %(trn_loss, trn_acc))
        print("val_loss=%.3f, val_acc=%.3f" %(val_loss, val_acc))

        # record & save model
        record(model, [alg, nodes1, nodes2, epoch, uniqIdx, norm, trn_loss, val_loss, trn_acc, val_acc])
        save_model(model, alg + '_' + str(nodes1) + '_' + str(nodes2) + '_' + str(epoch))

