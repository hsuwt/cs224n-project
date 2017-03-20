import pretty_midi
import numpy as np
import glob
import os
import util
import copy



def getMetadata(pm):
    # Extract informative events from the MIDI file
    return {'n_instruments': len(pm.instruments),
            'program_numbers': [i.program for i in pm.instruments if not i.is_drum],
            'key_numbers': [k.key_number for k in pm.key_signature_changes],
            'tempos': list(pm.get_tempo_changes()[1]),
            'time_signature_changes': pm.time_signature_changes,
            'end_time': pm.get_end_time(),
            'lyrics': [l.text for l in pm.lyrics]}

def labelmidi(path):
    try:
        pm = pretty_midi.PrettyMIDI(path)
        metadata = getMetadata(pm)
    except Exception as e:
        print(e)
        return False, []
    #print "loaded"
    estimated_tempo = 0
    # if change time_signature or no time_signature, skip this MIDI
    if len(metadata['time_signature_changes']) != 1:
        #print "no time signature"
        #estimated_tempo = pm.estimate_tempo()
        return False, []
    # if there are tempo changes, skip this MIDI
    elif len(metadata['tempos']) != 1:
        #print "tempo changes"
        return False, []
    # if there are less than 2 tracks, skip this MIDI
    if metadata['n_instruments'] < 2:
        #print "one track"
        return False, []
    # since there are many MIDIs without key label, we won't pre-process them.
    # Thus, we should be able to accompany songs with key changes
    tempo = 0
    if estimated_tempo == 0:
        tempo = metadata['tempos'][0]
    else:
        tempo = estimated_tempo
    fs = tempo/15 #1/(time per 16 beat)
    #print path, "is available midi with", len(pm.instruments), "instruments. Tempo=", tempo

    instrument_num = len(pm.instruments)
    ins_score = np.zeros(instrument_num)
    ismelody = np.ones(instrument_num, dtype=float)*100#0 for definitely not, -1 for bass, -2 for drum
    pitchrange = np.zeros(instrument_num)
    pitchmean = np.ones(instrument_num, dtype=float)*128
    pitchdeltasum = np.zeros(instrument_num)
    pitchdeltacount = np.zeros((instrument_num, 128))
    notetimesum = np.zeros(instrument_num)
    notequantize = np.zeros((instrument_num, 49))
    noteoverlap_2 = np.zeros(instrument_num)
    beats = 0

    #chord = set()
    #melody = set()
    for i in xrange(len(pm.instruments)):
        instrument = pm.instruments[i]
        #print "instrument", i
        if instrument.is_drum == True:
            ismelody[i] = -2
            continue

        piano_roll = instrument.get_piano_roll(fs=fs)#piano_roll
        beats = piano_roll.shape[1]
        lowest_pitch = 0
        highest_pitch = 0
        for p in xrange(128):#lowest pitch
            if np.count_nonzero(piano_roll[p]):
                lowest_pitch = p
                break
        for p in reversed(xrange(128)):#highest pitch
            if np.count_nonzero(piano_roll[p]):
                highest_pitch = p
                break
        pitchrange[i] = highest_pitch - lowest_pitch
        #print "pitch range = ", pitchrange[i],


        notecount = 0
        pitchsum = 0
        overlapcount = 0
        for b in xrange(beats):
            for p in xrange(128):
                if piano_roll[p][b]:
                    notecount += 1
                    pitchsum += p
            if np.count_nonzero(piano_roll[:, b])>=3:#three notes overlaps
                overlapcount += np.count_nonzero(piano_roll[:, b])
            if np.count_nonzero(piano_roll[:, b])==2:#two notes overlaps
                noteoverlap_2[i] += 1
            if any(piano_roll[:, b]):
                notetimesum[i] += 1
        if notecount:#if there is note -10
            pitchmean[i] = pitchsum/notecount
        else:
            ismelody[i] = -10
            break
        #print "mean pitch = ", pitchmean[i], "notes:", notecount,
        #print "total beats:", notetimesum[i], "(", notetimesum[i]*100/beats, "%)"

        if overlapcount/float(notecount) > 0.1:#total overlaps should be under 10% -11
            ismelody[i] = -11
        if notecount > notetimesum[i]*2: # too many notes -12
            ismelody[i] = -12
        if notetimesum[i]/beats<0.3:#play time under 30% -13
            ismelody[i] = -13
        if pitchrange[i] == 0:#same pitch ever -14
            ismelody[i] = -14


        if ismelody[i]>0:
            prevNoteStartTime = 0
            prevNoteEndTime = 0

            lastpitch = int(pitchmean[i])
            for note in instrument.notes:#notes lined by end time
                #print note, "    ", note.start - prevNoteStartTime

                notedelta = note.pitch-lastpitch
                #print notedelta
                pitchdeltasum += abs(notedelta)
                pitchdeltacount[i][abs(notedelta)] += 1
                lastpitch = note.pitch
                notedis = note.start - prevNoteStartTime
                noterest = note.start - prevNoteEndTime
                notelength = note.end - note.start
                notelengthquantize = int(np.round(notelength*fs))
                if notelengthquantize > 48:# note longer than 3 measures -15
                    ismelody[i] = -15
                else:
                    notequantize[i][notelengthquantize] += 1
                if ismelody[i] > 0:
                    if notedelta > 12:#score down if notes cross 12 pitch
                        ismelody[i] -= (notedelta-12)*100/notecount
                        if ismelody[i]<1:
                            ismelody[i]=1
                else:
                    break

                prevNoteStartTime = note.start
                prevNoteEndTime = note.end
            #if ismelody[i]:
            #    print pitchdeltacount[i]
            else: continue


        #util.writecsv(piano_roll, "piano_roll.csv")
        #break
    bassflag = True
    for ba in xrange(instrument_num):
        if 33<= pm.instruments[ba].program <=40 or pitchmean[ba] <= 40:
            ismelody[ba] = -1
            bassflag = False
    if bassflag:
        if np.min(pitchmean)<=50:
            ismelody[np.argmin(pitchmean)] = -1
    if any(ismelody):
        m = max(ismelody)
        maxindex = [ss for ss, s in enumerate(ismelody) if s == m]
        if len(maxindex) > 1:
            #print "compare overlaps"
            for mi in maxindex:
                #print noteoverlap_2[mi]/beats
                ismelody[mi] *= (1-noteoverlap_2[mi]/beats)#less overlaps
        m = max(ismelody)
        maxindex = [ss for ss, s in enumerate(ismelody) if s == m]
        if len(maxindex) > 1:
            for z in xrange(instrument_num):
                if z not in maxindex:
                    ismelody[z] = 0
            #print "compare note length"
            for mi in maxindex:
                #print notequantize[mi][0:16]
                if notequantize[mi][0]+notequantize[mi][1] > notequantize[mi][2]: #16 beats more than 8 beats
                    ismelody[mi] *= 0.8
        m = max(ismelody)
        maxindex = [ss for ss, s in enumerate(ismelody) if s == m]
        if len(maxindex) > 1:
            #print "compare total time"
            for mi in maxindex:
                #compare playing time - problem with guitar
                ismelody[mi] += pitchrange[mi]
    #print "########################################################"
    """
    clonepm = copy.deepcopy(pm)
    if any(ismelody>0):
        print np.argmax(ismelody), ismelody.astype(int)
        clonepm.instruments = [clonepm.instruments[np.argmax(ismelody)]]
    else:
        clonepm.instruments = [clonepm.instruments[4]]

    clonepm.write('temp.mid')
    #util.play_midi('temp.mid')

    #"""
    #break
    if any(ismelody>0):
        #print np.argmax(ismelody), ismelody.astype(int)
        return True, ismelody
    else:
        return False, []

    # TODO: try to get the minimal time unit, and parse MIDI into song structure.


    # Here are a list of functions to get the beats, downbeats, onset information
    # Although it's not matched to the exact time, I think using the prevNote method above is better!
    #http://craffel.github.io/pretty-midi/
    # print pm.get_onsets()
    # print pm.get_downbeats()
    # print pm.get_beats()
    # pm.estimate_beat_start()
def labeltocsv(directory, path, label):
    pm = pretty_midi.PrettyMIDI(directory+path)
    metadata = getMetadata(pm)
    tempo = metadata['tempos'][0]
    fs = tempo/15
    piano_roll = pm.get_piano_roll(fs=fs)#piano_roll
    beats = piano_roll.shape[1]
    end = pm.get_end_time()
    melody = np.zeros((128, beats))
    chord = np.zeros((128, beats))
    bass = np.zeros((128, beats))

    for i in xrange(len(pm.instruments)):
        if label[i] == np.max(label):
            melody = pm.instruments[i].get_piano_roll(times=np.arange(0, end, 1./fs))
        elif label[i] == -1:
            tmp = pm.instruments[i].get_piano_roll(times=np.arange(0, end, 1./fs))[:, :beats]
            if tmp.shape == bass.shape:
                bass += tmp
        elif label[i] == -2:
            pass
        else:
            tmp = pm.instruments[i].get_piano_roll(times=np.arange(0, end, 1./fs))[:, :beats]
            if tmp.shape == chord.shape:
                chord += tmp

    util.writecsv(melody, "midilabel/melody/"+path[:-3]+"csv")
    util.writecsv(chord, "midilabel/chord/"+path[:-3]+"csv")
    util.writecsv(bass, "midilabel/bass/"+path[:-3]+"csv")
