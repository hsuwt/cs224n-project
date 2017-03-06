
import pretty_midi
import math
import csv
import numpy as np
from util import *
from os import listdir
from os.path import isfile, join
from scipy import stats

if __name__ == "__main__":
    # get the list of pathname in the directory TheoryTab
    path = 'MIDI_MSD/'
    onlyfiles = [ f for f in listdir(path) if isfile(join(path,f))]
    # initialize
    song = [] # List of song names
    errorSong = []
    errorCnt = [0] * len(onlyfiles)

    numType = 5 # maj, min, maj7, 7, min7
    numSong = len(onlyfiles)
    melody = np.zeros((numSong, 12, 128), dtype = int)
    chord  = np.zeros((numSong, 12, 128), dtype = int)
    root   = np.zeros((numSong, 12, 128), dtype = int)

    for i in range(len(onlyfiles)):
        song.append(onlyfiles[i][0:-4].split(';'))
        song[i].append(path + onlyfiles[i])
        midi = pretty_midi.PrettyMIDI(song[i][-1]) # load MIDI file
        for k in range(len(midi.instruments)):
            notes = midi.instruments[k].notes # all notes in the i-th song
            #unit = (notes[-1].end)/128
            chord_list = isChord(notes)
            if chord_list:
                print("\n%dth song, %dth instrument are all chords" %(i, k))
                for chord in chord_list:
                    chroma = np.zeros((1,12,1))
                    for j in range(len(chord)):
                        chroma[0][chord[j].pitch%12][0] = 1
                    r, t = chroma2chord_v2(chroma[0][:][:,0])
                    print('%d notes in a %s%s,\t t=[%0.3f, %0.3f]' %(len(chord), int2note(r), int2type(t), chord[0].start, chord[0].end))
