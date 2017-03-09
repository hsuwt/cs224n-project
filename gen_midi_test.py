import util
import numpy as np


midi_output_path = 'midi_out/'
mel = np.genfromtxt('melody_test.csv', delimiter=',')

chords_ground = np.genfromtxt('pred/chord_test.csv', delimiter = ',')
#chords_LM = np.genfromtxt('pred/pred_LM.csv', delimiter = ',')
chords_pair_L1diff = np.genfromtxt('pred/pred_pair_L1diff.csv', delimiter = ',')
chords_pair = np.genfromtxt('pred/pred_pair.csv', delimiter = ',')


for i in range(10):
    song_ground = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_ground[128*i:128*(i+1)])
    #song_LM = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_LM[128*i:128*(i+1)])
    song_pair = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_pair[128*i:128*(i+1)])
    song_pair_L1diff = util.Matrices_to_MIDI(mel[128*i:128*(i+1)],chords_pair[128*i:128*(i+1)])

    song_ground.write(midi_output_path+str(i)+'_ground.mid')
    #song_LM.write(midi_output_path+str(i)+'_LM.mid')
    song_pair.write(midi_output_path+str(i)+'_pair.mid')    
    song_pair_L1diff.write(midi_output_path+str(i)+'_pair_L1diff.mid')    

#print np.sum(np.abs(chords_ground[:128] - chords_LM[:128]))
#print np.sum(np.abs(chords_ground[:128] - chords_pair[:128]))
