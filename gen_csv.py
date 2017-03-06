
import pretty_midi
import math
import csv
import numpy as np
import os
from scipy import stats
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
keys = []
modes = []
bars = []
skip = []
for csvfile in csvfiles:
    with open(csvfile, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        rows = iter(reader)
        next(rows)
        for row in rows:
            key, mode = toMajKey(note2int(row[5]), mode2int(row[6]))
            keys.append(key)
            modes.append(mode)
            bars.append(int(row[-1]))
assert len(keys) == len(modes) == len(midfiles)

# initialize
numSong = len(midfiles)
numType = 5 # maj, min, maj7, 7, min7
errorSong = []
errorCnt = [0] * numSong
melody = np.zeros((numSong, 12, 128), dtype = int)
chord  = np.zeros((numSong, 12, 128), dtype = int)
root   = np.zeros((numSong, 12, 128), dtype = int)
R = np.zeros((numSong, 128), dtype = int)
M = np.zeros((numSong, 128), dtype = int)

for i in range(len(midfiles)):
    if bars[i] > 8: continue
    midi = pretty_midi.PrettyMIDI(midfiles[i]) # load MIDI file
    unit = midi.instruments[1].notes[-1].end/128 # the smallest time unit
    for k in range(len(midi.instruments)):
        notes = midi.instruments[k].notes # all notes in the i-th song
        for j in xrange(len(notes)):
            notes[j].start /= float(unit)
            notes[j].end   /= float(unit)

            # filter out mis-matched songs
            t1 = round((notes[j].start),2)
            t2 = round((notes[j].end),2)
            diff1 = t1-math.floor(t1)
            diff2 = t2-math.floor(t2)
            thres = 0.3
            if (diff1 < 1-thres and diff1 > thres) or (diff2 < 1-thres and diff2 > thres) :
                errorCnt[i] += 1

            # save data of melody, chord, and root
            note = (notes[j].pitch - keys[i]) % 12
            idx1 = int(round(t1))
            idx2 = int(round(t2))
            if k==0: melody[i][note][idx1:idx2] = np.ones((idx2-idx1))
            if k==1:  chord[i][note][idx1:idx2] = np.ones((idx2-idx1))
            if k==2:   root[i][note][idx1:idx2] = np.ones((idx2-idx1))

for i in list(reversed(range(len(errorCnt)))):
    # most of the errors come from the root
    # a few errors come from the melody
    # no errors come from the chord
    if errorCnt[i] > 40:
        print midfiles[i], 'err=', errorCnt[i], 'bars=', bars[i]
        melody = np.delete(melody, (i), axis=0)
        chord  = np.delete(chord,  (i), axis=0)
        root   = np.delete(root,   (i), axis=0)
        del errorCnt[i]

numSong = len(melody)
template        = np.zeros((12*numType,12), dtype=int)
templateMerged  = np.zeros((numType,12), dtype=int)
templateNorm    = np.zeros((numType,12), dtype=float)

for i in range(numSong):
    for j in range(128):
        r = np.nonzero(  root[i][:][:,j])[0]
        m = np.nonzero(melody[i][:][:,j])[0]
        if len(r)==1: # many len(r) equals 0, meaning that many instances are not counted
            chord[i][r[0]][j] = 1
        r,t = chroma2chord_v2(chord[i][:][:,j])
        if t not in [1,2,3,4,5]: t=0
        R[i][j] = r
        if len(m)==1 and t:
            M[i][j] = m[0]
            template[(t-1)*12+r][m[0]] += 1

for i in range(12):
    for j in range(12):
        for k in range(numType):
            templateMerged[k][i] += template[j+12*k][(i+j)%12]

# normalize the template
templateNorm = stats.zscore(templateMerged, axis=1)
templateNorm[np.isnan(templateNorm)] = 0

melody = melody.reshape((numSong,12*128))
chord  = chord .reshape((numSong,12*128))
root   = root  .reshape((numSong,12*128))

melody.tolist()
chord.tolist()
root.tolist()
R.tolist()
templateMerged.tolist()
templateNorm.tolist()

with open('csv/melody.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(M)
with open('csv/chord.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(chord)
with open('csv/root.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(R)
with open('csv/template.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(templateMerged)
with open('csv/templateN.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(templateNorm)
