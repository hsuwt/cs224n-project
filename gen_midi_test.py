import util
import numpy as np

from os import listdir
from os.path import isfile, join
mypath = 'pred/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

midi_output_path = 'midi_out/'
mel = np.genfromtxt('melody_test.csv', delimiter=',')
chords_ground = np.genfromtxt('chord_test.csv', delimiter = ',')

for i in range(10):
    song_ground = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_ground[128*i:128*(i+1)])
    song_ground.write(midi_output_path+str(i)+'_ground.mid')
    for file in files:
		chords_pred = np.genfromtxt('pred/' + file, delimiter = ',')
		song_pred = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_pred[128*i:128*(i+1)])
		song_pred.write(midi_output_path+str(i)+'_'+file+'.mid')
