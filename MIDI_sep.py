
import pretty_midi
import csv
import os
import copy
import wave
from util import *

# get the list of midfiles and csvfiles
path = 'MIDI/'
midfiles = []
csvfiles = []
for file in os.listdir(path):
    if file.endswith('.mid'): midfiles.append(path+file)
    if file.endswith('.csv'): csvfiles.append(path+file)
midfiles.sort()
csvfiles.sort()

# record the key and mode in csvfiles
bars = []
for csvfile in csvfiles:
    with open(csvfile, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        rows = iter(reader)
        next(rows)
        for row in rows:
            bars.append(int(row[-1]))

new_path = ['MIDI_sep/melody/', 'MIDI_sep/chord/', 'MIDI_sep/root/']
for i in range(len(midfiles)):
    if bars[i] > 8: continue
    midi = pretty_midi.PrettyMIDI(midfiles[i]) # load MIDI file
    for k in range(len(midi.instruments)):
        midi_sep = copy.deepcopy(midi)
        midi_sep.instruments[0] = midi_sep.instruments[k]
        del midi_sep.instruments[1:3]
        midi_sep.write(new_path[k]+str(i)+'.mid')
        audio = midi.synthesize()

