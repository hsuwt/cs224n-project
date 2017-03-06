import os
import csv
from keras.layers import Input, Dense, Dropout, Reshape, Permute, merge, Flatten
from keras.layers import Convolution2D, Convolution3D, ZeroPadding2D, ZeroPadding3D
from keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed
from keras.models import Model, model_from_json
from keras.optimizers import RMSprop, Adagrad
from keras.callbacks import EarlyStopping

modelpath = 'model2/'

def load_model(alg):
    model = model_from_json(open(modelpath + alg + '.json').read())
    model.load_weights(modelpath + alg + '.h5')
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_model(model, alg):
    open(modelpath + alg + '.json', 'w').write(model.to_json())
    model.save_weights(modelpath + alg + '.h5', overwrite=True)

def gen_input(feat):
    dim = 12
    if feat == 'pair':
        dim = 12 * 2
    return Input(shape=(128, dim))

def build(alg, input, nodes, drp):
    if 'DNN' in alg:
        M = Flatten()(input)
        M = Dense(nodes, activation='relu')(M)
        M = Dropout(drp)(M)
        return M
    elif 'RNN' in alg or 'LSTM' in alg or 'GRU' in alg:
        return_sequences = False
        if '128' in alg or 'baseline' in alg: return_sequences = True
        if 'RNN' in alg:
            M1 = SimpleRNN(nodes, return_sequences=return_sequences)(input)
            M2 = SimpleRNN(nodes, return_sequences=return_sequences)(input)
        elif 'LSTM' in alg:
            M1 = LSTM(nodes, return_sequences=return_sequences)(input)
            M2 = LSTM(nodes, return_sequences=return_sequences)(input)
        elif 'GRU' in alg:
            M1 = GRU(nodes, return_sequences=return_sequences)(input)
            M2 = GRU(nodes, return_sequences=return_sequences)(input)
        M1 = Dropout(drp)(M1)
        M2 = Dropout(drp)(M2)
        if 'B' in alg:
            M = merge([M1, M2], mode='concat')
            return M
        return M1

def build_model(alg, nodes1, nodes2, drp):
    if 'baseline' in alg:
        input = gen_input('melody')
        M = build(alg, input, nodes1, drp)
        if 'RNN' in alg or 'LSTM' in alg or 'GRU' in alg:
            output = TimeDistributed(Dense(12, activation='sigmoid'))(M)
        else:
            M = Dense(128 * 12, activation='sigmoid')(M)
            output = Reshape((128, 12))(M)
        model = Model(input=input, output=output)
    elif 'LM' in alg:
        input = gen_input('melody')
        M = build(alg, input, nodes1, drp)
        if 'RNN' in alg or 'LSTM' in alg or 'GRU' in alg:
            output = TimeDistributed(Dense(12, activation='sigmoid'))(M)
    elif 'pair' in alg:
        input = gen_input('pair')
        M = build(alg, input, nodes1, drp)
        if '128' in alg:
            output = TimeDistributed(Dense(1, activation='sigmoid'))(M)
        else:
            output = Dense(1, activation='sigmoid')(M)
        model = Model(input=input, output=output)
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def record(model, rec):
    return
    header = ["alg", "nodes1", "nodes2", "epoch", "uniqIdx", "hamDis", "trn_loss", "val_loss", "trn_acc", "val_acc"]
    alg      = rec[0]
    nodes1   = rec[1] = int(rec[1])
    nodes2   = rec[2] = int(rec[2])
    epoch    = rec[3] = int(rec[3])
    uniqIdx  = rec[4] = int(rec[4])
    hamDis   = rec[5] = round(float(rec[5]), 2)
    trn_loss = rec[6] = round(float(rec[6]), 2)
    val_loss = rec[7] = round(float(rec[7]), 2)
    trn_acc  = rec[8] = round(float(rec[8]), 2)
    val_acc  = rec[9] = round(float(rec[9]), 2)

    # append the new rec to rec.csv
    with open('record/rec.csv', 'rb') as csvfile:
        mat = [row for row in csv.reader(csvfile)]
    del mat[0]
    if len(mat):
        mat = map(list, zip(*mat)) # transpose
        mat[1] = [int(i) for i in mat[1]] # nodes1
        mat[2] = [int(i) for i in mat[2]] # nodes2
        mat[3] = [int(i) for i in mat[3]] # epoch
        mat[4] = [int(i) for i in mat[4]] # uniqIdx
        mat[5] = [round(float(i), 2) for i in mat[5]] # hamDis
        mat[6] = [round(float(i), 2) for i in mat[6]] # trn_loss
        mat[7] = [round(float(i), 2) for i in mat[7]] # val_loss
        mat[8] = [round(float(i), 2) for i in mat[8]] # trn_acc
        mat[9] = [round(float(i), 2) for i in mat[9]] # val_acc
        mat = map(list, zip(*mat)) # transpose
    mat.append(rec)
    mat.sort()
    mat.insert(0, header)
    with open('record/rec.csv', 'wb') as csvfile:
        csv.writer(csvfile, lineterminator=os.linesep).writerows(mat)

    # check whether it beats the best record
    for eval in ['hamDis', 'uniqIdx', 'loss', 'acc']:
        flag = False
        with open('record/' + eval + '.csv', 'rb') as csvfile:
            mat = [row for row in csv.reader(csvfile)]
        del mat[0]
        if len(mat):
            for i in range(len(mat)):
                if mat[i][0] == alg:
                    flag = True
                    if eval == 'hamDis' and hamDis < float(mat[i][5]) or eval == 'uniqIdx' and uniqIdx > int(mat[i][4]) \
                    or eval == 'loss' and val_loss < float(mat[i][7]) or eval == 'acc' and val_acc < float(mat[i][9]):
                        mat[i] = rec
                        with open('record/' + eval + '.csv', 'w') as csvfile:
                            mat.insert(0, header)
                            csv.writer(csvfile, lineterminator=os.linesep).writerows(mat)
                        save_model(model, alg + '_' + eval)
                        if eval == 'hamDis':
                            save_model(model, alg)
                    break
        if not flag: # It's the first alg
            with open('record/' + eval + '.csv', 'a') as csvfile:
                csv.writer(csvfile, lineterminator=os.linesep).writerow(rec)
            save_model(model, alg + '_' + eval)
