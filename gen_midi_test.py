import util
import numpy as np
from argparse import ArgumentParser

from os import listdir, mkdir
from os.path import isfile, join, exists

parser = ArgumentParser()
parser.add_argument('number', type=int, default=10)
parser.add_argument('pred_path', nargs='?', default='../pred/', help="Path prediction files (csv), defaults to ../pred")
parser.add_argument('midi_output_path', nargs='?', default='../midi_out/',
                    help="Path to output midi files, defaults to ../midi_out")
parser.add_argument('nb_train', nargs='?', type=int, default=1000)
args = parser.parse_args()


def _ensure_is_good_path(path):
    if not path.endswith('/'):
        return path + '/'
    else:
        return path

pred_path = _ensure_is_good_path(args.pred_path)
files = [f for f in listdir(pred_path) if f.endswith('.npy')]
midi_output_path = _ensure_is_good_path(args.midi_output_path)
if not exists(midi_output_path):
    print 'Can\'t find %s, creating new directory'
    mkdir(midi_output_path)

M = np.load('csv/normal-melody.npy')
C = np.load('csv/normal-chord.npy')

nb_train = args.nb_train
M = M[-nb_train:]
C = C[-nb_train:]

for f in files:
    print f
    for i in range(args.number):
        C_pred = np.load('../pred/' + f)
        song_pred = util.matrices2midi(M[i], C_pred[i])
        song_pred.write(midi_output_path + str(i) + '_' + f[:-4] + '.mid')

print "Reconstructing midi files of ground truth"
for i in range(args.number):
    song_ground = util.matrices2midi(M[i], C[i])
    song_ground.write(midi_output_path + str(i) + '_ground.mid')