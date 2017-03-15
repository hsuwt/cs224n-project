import util
import numpy as np

from os import listdir
from os.path import isfile, join
mypath = '../pred/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
midi_output_path = 'midi_out/'
M = np.load('../npy/melody_csv1.npy')
C = np.load('../npy/chord_csv1.npy')
nSong = 1000
M = M[-nSong:]
C = C[-nSong:]

for i in range(10):
    song_ground = util.Matrices_to_MIDI(M[i], C[i])
    song_ground.write(midi_output_path + str(i) + '_ground.mid')
    for file in files:
        C_pred = np.load('../pred/' + file)
	song_pred = util.Matrices_to_MIDI(M[i], C_pred[i])
	song_pred.write(midi_output_path + str(i) + '_' + file[:-4] + '.mid')


