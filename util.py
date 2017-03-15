import pickle as pkl
import sys
import os
import numpy as np
import pretty_midi
from build_chord_repr import ChordNotes2OneHotTranscoder, get_onehot2chordnotes_transcoder

def parse_algorithm(alg_str):
    alg = {x: None for x in alg_str.strip().split()}
    if 'one-hot' in alg:
        alg['one-hot-dim'] = 0  # to be filled in
    return alg


def rotate(_chroma, semitone):
    if semitone == 0:
        return _chroma
    return np.concatenate((_chroma[-semitone:], _chroma[:_chroma.shape[0] - semitone]), axis=0)


def hamDis(chroma1, chroma2):
    assert chroma1.shape == chroma2.shape
    return float(np.count_nonzero(chroma1 != chroma2))


def union(chroma1, chroma2):
    assert chroma1.shape[0] == 12
    assert chroma2.shape[0] == 12
    ret = np.zeros(12)
    for i in range(12):
        if chroma1[i] and chroma2[i]:
            ret[i] = 1
    return ret

chroma2chord_LUT = np.array([
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
])


def closestChord(root, chordType):
    return rotate(chroma2chord_LUT[chordType].copy(), root)


def chroma2chord_v2(_chroma):
    # v2 is way faster
    notes = np.where(_chroma)[0]
    interval = np.diff(notes)
    if len(notes) == 3:
        if all(interval==np.array([4,3])): return [notes[0],1] #maj root position
        if all(interval==np.array([3,5])): return [notes[2],1] #maj 1st inversion
        if all(interval==np.array([5,4])): return [notes[1],1] #maj 2nd inversion
        if all(interval==np.array([3,4])): return [notes[0],2] #maj root position
        if all(interval==np.array([4,5])): return [notes[2],2] #maj 1st inversion
        if all(interval==np.array([5,3])): return [notes[1],2] #maj 2nd inversion
        return [0,6]
    if len(notes) == 4:
        if all(interval==np.array([4,3,4])): return [notes[0],3] #maj7 root position
        if all(interval==np.array([3,4,1])): return [notes[3],3] #maj7 1st inversion
        if all(interval==np.array([4,1,4])): return [notes[2],3] #maj7 2nd inversion
        if all(interval==np.array([1,4,3])): return [notes[1],3] #maj7 3rd inversion
        if all(interval==np.array([4,3,3])): return [notes[0],4] #dmn7 root position
        if all(interval==np.array([3,3,2])): return [notes[3],4] #dmn7 1st inversion
        if all(interval==np.array([3,2,4])): return [notes[2],4] #dmn7 2nd inversion
        if all(interval==np.array([2,4,3])): return [notes[1],4] #dmn7 3rd inversion
        if all(interval==np.array([3,4,3])): return [notes[0],5] #min7 root position
        if all(interval==np.array([4,3,2])): return [notes[3],5] #min7 1st inversion
        if all(interval==np.array([3,2,3])): return [notes[2],5] #min7 2nd inversion
        if all(interval==np.array([2,3,4])): return [notes[1],5] #min7 3rd inversion
        return [0,6]
    if len(notes) == 2:
        if all(interval==np.array([7])): return [notes[0],7] #power chord
        if all(interval==np.array([5])): return [notes[1],7] #power chord
        return [0,6]
    return [0,6]


def chroma2chord_v1(chroma):
    for i in range(12):
        chroma_shifted = rotate(chroma, i)
        if all(chroma_shifted == np.array([1,0,0,0,1,0,0,1,0,0,0,0])): return [(12-i)%12,1] #maj
        if all(chroma_shifted == np.array([1,0,0,1,0,0,0,1,0,0,0,0])): return [(12-i)%12,2] #min
        if all(chroma_shifted == np.array([1,0,0,0,1,0,0,1,0,0,0,1])): return [(12-i)%12,3] #maj7
        if all(chroma_shifted == np.array([1,0,0,0,1,0,0,1,0,0,1,0])): return [(12-i)%12,4] #7
        if all(chroma_shifted == np.array([1,0,0,1,0,0,0,1,0,0,1,0])): return [(12-i)%12,5] #min7
    return [0,0]


def isChord(notes):
    if len(notes) < 2:
        return False
    chord_list = []
    while len(notes) >= 2:
        if notes[0].start != notes[1].start or notes[0].end != notes[1].end:
            return False
        chord = []
        while len(notes) >= 2 and notes[0].start == notes[1].start and notes[0].end == notes[1].end:
            chord.append(notes[0])
            del notes[0]
        chord.append(notes[0])
        del notes[0]
        chord_list.append(chord)
    # assert len(notes) == 0
    return chord_list

_rootNote = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#',
    4: 'E', 5: 'F', 6: 'F#', 7: 'G',
    8: 'G#', 9: 'A', 10: 'A#', 11: 'B',
}

_qualifier = {
    1: 'Maj', 2: 'Min', 3: 'Maj7', 4: '7', 5: 'Min7'
}


def printChord(_chroma):
    r, t = chroma2chord_v2(_chroma)
    S = _rootNote[r] + _qualifier[t] if t != 6 else 'N'
    print '%s\t' % S


def printChordProgression(y, cp):
    assert y.shape == cp.shape
    assert y.shape[1] == 128
    assert y.shape[2] == 12

    n_song = len(cp)
    for i in range(n_song):
        print 'song %d\t answ:' % i
        for j in range(8):
            printChord(y[i][16*j])
        print '\nsong %d\t pred:' % i
        for j in range(8):
            printChord(cp[i][16*j])
        print '\nsong %d\t diff:' % i
        cnt = 0
        for j in range(8):
            tmp = np.sum(abs(y[i][16*j]-cp[i][16*j]))
            cnt += tmp
            print '%d\t' % tmp
        print '%d\n' % cnt


def pred2chord(pred):
    pred = pred.reshape((pred.shape[0], 12, 8, pred.shape[1]/12/8))
    pred = np.mean(pred, axis=3)
    for i in range(len(pred)):
        for j in range(len(pred[0][0])):
            pred[i][:][:,j] = closestChord(pred[i][:][:,j])


def toCandidate(CP, allCP, bestN, criteria):
    ret = np.zeros_like(CP)
    bestIdx = np.zeros((len(CP), bestN), dtype=int)
    for i in range(len(CP)):
        minDis = sys.maxint
        minIdx = 0
        dis = np.zeros((len(allCP)))
        for j in range(len(allCP)):
            if criteria == 'L1':
                dis[j] = np.sum(abs(CP[i]-allCP[j]))
            elif criteria == 'L2':
                dis[j] = np.sqrt(np.sum(np.square(CP[i]-allCP[j])))
            else:
                print("Error in toCandidate function")
            if dis[j] < minDis:
                minDis = dis[j]
                minIdx = j
        ret[i] = allCP[minIdx]
        bestIdx[i] = np.argsort(dis)[:bestN]
    return ret, bestIdx


def toCandidateBestN(CP, allCP, bestN):
    bestIdx = np.zeros((len(CP), bestN), dtype=int)
    for i in range(len(CP)):
        dis = np.zeros((len(allCP)))
        for j in range(len(allCP)):
            dis[j] = np.sum(abs(CP[i]-allCP[j]))
        bestIdx[i] = np.argsort(dis)[:bestN]
    return bestIdx



def parse_data(alg, max_length):
    if 'pair' in alg:
        C = np.genfromtxt('csv/chord.csv', delimiter=',')
        # Data in melody.csv and root.csv are represented as [0,11].
        # Thus, we first span it to boolean matrix
        M_dense = np.genfromtxt('csv/melody.csv', delimiter=',')
        assert (M_dense.shape[1]*12 == C.shape[1])
        M = np.zeros((M_dense.shape[0], M_dense.shape[1]*12))
        sample_weight = np.zeros(M_dense.shape)
        for i in range(M_dense.shape[0]):
            for j in range(M_dense.shape[1]):
                if not np.isnan(M_dense[i][j]):
                    notes = int(M_dense[i][j])
                    M[i][M_dense.shape[1]*notes+j] = 1
                    sample_weight[i][j] = 1
        C = np.nan_to_num(C)
        M = np.swapaxes(M.reshape((M_dense.shape[0],12,M_dense.shape[1])), 1, 2)
        C = np.swapaxes(C.reshape((C.shape[0],12,-1)), 1, 2)
        return C, M, np.ones((C.shape[0], C.shape[1]))
    #elif 'LM' in alg:
    elif True:
        nb_train = sum([len(files) for r, d, files in os.walk("../dataset/melody")])
        C = np.zeros((nb_train, max_length, 12))
        M = np.zeros((nb_train, max_length, 12))
        sample_weight = np.zeros((nb_train, max_length))
        train_idx = 0
        for root,_,files in os.walk("../dataset/melody"):
            for m_file_name in files:
                m_file_name_path = os.path.join(root, m_file_name)
                c_file_name_path = m_file_name_path.replace("/melody/", "/chord/", 1)
                m_file_matrix = np.genfromtxt(m_file_name_path, delimiter=',')
                c_file_matrix = np.genfromtxt(c_file_name_path, delimiter=',')
                if len(m_file_matrix) == 0 or len(c_file_matrix) == 0: continue
                m_file_matrix = m_file_matrix[:,:max_length]
                c_file_matrix = c_file_matrix[:,:max_length]
                seq_len = min(m_file_matrix.shape[1], c_file_matrix.shape[1])
                tmp_m, tmp_c = np.zeros((12,seq_len)), np.zeros((12,seq_len))
                for i in range(128/12):
                    tmp_m = np.logical_or(tmp_m, m_file_matrix[i*12:(i+1)*12,:seq_len] )
                    tmp_c = np.logical_or(tmp_c, c_file_matrix[i*12:(i+1)*12,:seq_len] )
                sample_weight[train_idx,:seq_len] = np.ones((1,seq_len))
                M[train_idx,:seq_len], C[train_idx,:seq_len] = np.swapaxes(tmp_m, 0,1), np.swapaxes(tmp_c,0,1)
                train_idx +=1
        C = C[:train_idx]
        M = M[:train_idx]
        sample_weight = sample_weight[:train_idx]
        #np.save('csv1.npy', np.stack((C, M, sample_weight)))
        return C, M, sample_weight
    else: assert False


def load_data(alg, nb_test):
    max_length = 1024
    C, M, sample_weight = parse_data(alg, max_length)
    m = M[-nb_test:]
    c = C[-nb_test:]
    sw = sample_weight[-nb_test:]
    M = M[:-nb_test]
    C = C[:-nb_test]
    SW = sample_weight[:-nb_test]
    return M, m, C, c, SW, sw


class InputParser(object):
    def __init__(self, alg):
        if 'LM' in alg and 'one-hot' in alg:
            self.transcoder = ChordNotes2OneHotTranscoder()
        self.alg = alg

    def get_XY(self, M, C):
        if 'LM' in self.alg and 'one-hot' in self.alg:
            """
            the dim of chord (C or Y) will change from 12 into {self.size}
            """
            C = self.transcoder.transcode(C)
            return M, C

        assert 'pair' in self.alg
        n = M.shape[0]
        idx = np.random.randint(n, size=n)
        C_neg = C[idx]
        Ones = np.ones((n, 128, 1))
        Zeros = np.zeros((n, 128, 1))
        if 'L1' in self.alg or 'L2' in self.alg or 'L1diff' in self.alg:
            # use L1 or L2 of two sources of chord as labels
            np.seterr(divide='ignore', invalid='ignore')
            L1diff = (C_neg - C) / 2 + 0.5
            L1 = np.sum(abs(C - C_neg), 2)
            L1 = L1.reshape((n, 128, 1))
            L2 = np.sqrt(L1)
            p = np.sum(np.logical_and(C, C_neg), 2) / np.sum(C_neg, 2)
            r = np.sum(np.logical_and(C, C_neg), 2) / np.sum(C, 2)
            F1 = 2*p*r/(p+r)
            F1 = np.nan_to_num(F1.reshape((n, 128, 1)))
            if 'rand' in self.alg:
                X = np.concatenate((M, C_neg), 2)
                Y = L1 if 'L1' in self.alg \
                    else L2 if 'L2' in self.alg \
                    else F1 if 'F1' in self.alg \
                    else L1diff
            else:
                MC_neg = np.concatenate((M, C_neg), 2)
                MC = np.concatenate((M, C), 2)
                X = np.concatenate((MC, MC_neg), 0)
                Y = np.concatenate((Zeros, L1), 0) if 'L1' in self.alg \
                    else np.concatenate((Zeros, L2), 0) if 'L2' in self.alg \
                    else np.concatenate((Ones, F1), 0) if 'F1' in self.alg \
                    else np.concatenate((np.tile(Zeros, 12) + 0.5, L1diff), 0)
            Y = 1 - Y / 12.0
        return X, Y

def get_test(alg, m, C):
    # x_te are the final testing features to match m to C
    if 'pair' in alg:
        m_rep, C_rep = rep(m, C)
        return np.concatenate((m_rep, C_rep), 2)
    elif 'LM' in alg:
        return m;
    else:
        print('Error in get_test')

def print_result(pred, y, Y, alg, printCP, bestN):
    print('\nAlg: %s' %(alg))
    nb_test = pred.shape[0]
    if 'L2' in alg:
        pred, bestNIdx = toCandidate(pred, Y, bestN, 'L2')
        norm = np.sqrt(np.sum(np.square(pred - y))) / 128.0 / nb_test
    else:
        pred, bestNIdx = toCandidate(pred, Y, bestN, 'L1')
        norm = np.sum(abs(pred - y)) / 128.0 / nb_test
    numUniqIdx = len(np.unique(bestNIdx))
    if printCP: printChordProgression(y, pred)
    print('num of unique idx  = %d/%d' %(numUniqIdx, nb_test))
    print('norm after mapping = %.3f' %(norm))
    return bestNIdx, numUniqIdx, norm

def rep(m, C):
    nb_test = m.shape[0]
    nb_train = C.shape[0]
    C_rep = np.tile(C, (nb_test,  1, 1))
    m_rep = np.tile(m, (nb_train, 1, 1))
    m_rep = np.reshape(m_rep, (nb_train, nb_test, 128, 12))
    m_rep = np.swapaxes(m_rep, 1, 0)
    m_rep = np.reshape(m_rep, (nb_test * nb_train, 128, 12))
    return m_rep, C_rep

def note2int(note):
    if note == 'C' : return 0
    if note == 'C#': return 1
    if note == 'Db': return 1
    if note == 'D' : return 2
    if note == 'D#': return 3
    if note == 'Eb': return 3
    if note == 'E' : return 4
    if note == 'F' : return 5
    if note == 'F#': return 6
    if note == 'Gb': return 6
    if note == 'G' : return 7
    if note == 'G#': return 8
    if note == 'Ab': return 8
    if note == 'A' : return 9
    if note == 'A#': return 10
    if note == 'Bb': return 10
    if note == 'B' : return 11
    else:
        print 'no match in note2int: ', note
        return 0

def mode2int(mode):
    mode = mode.title()
    if mode == 'Major'     : return 0
    if mode == 'Minor'     : return 1
    if mode == 'Dorian'    : return 2
    if mode == 'Phrygian'  : return 3
    if mode == 'Lydian'    : return 4
    if mode == 'Mixolydian': return 5
    if mode == 'Locryan'   : return 6
    else:
        print 'no match in mode2int: ', mode
        return 0

def int2note(i):
    if i == 0 : return 'C'
    if i == 1 : return 'C#'
    if i == 2 : return 'D'
    if i == 3 : return 'D#'
    if i == 4 : return 'E'
    if i == 5 : return 'F'
    if i == 6 : return 'F#'
    if i == 7 : return 'G'
    if i == 8 : return 'G#'
    if i == 9 : return 'A'
    if i == 10: return 'A#'
    if i == 11: return 'B'
    else:
        print 'no match in int2note: ', i
        return 'C'

def int2mode(i):
    if i == 0: return 'Major'
    if i == 1: return 'Minor'
    if i == 2: return 'Dorian'
    if i == 3: return 'Phrygian'
    if i == 4: return 'Lydian'
    if i == 5: return 'Mixolydian'
    if i == 6: return 'Locryan'
    else:
        print 'no match in int2mode: ', i
        return 'Major'

def int2type(i):
    if i == 0: return 'Non-chord'
    if i == 1: return 'Maj'
    if i == 2: return 'Min'
    if i == 3: return 'Maj7'
    if i == 4: return 'Dominant7'
    if i == 5: return 'Min7'
    if i == 6: return 'omplex' #complex chord, but not recognized
    if i == 7: return 'power' #power chord, but not recognized
    else:
        print 'no match in int2type: ', i
        return 'Non-chord'


def toMajKey(key, mode):
    if mode == 1: # Minor to Major
        key = (key + 3) % 12
    elif mode == 2: # Dorian to Major
        key = (key + 10) % 12
    elif mode == 3: # Phrygian to Major
        key = (key + 8) % 12
    elif mode == 4: # Lydian to Major
        key = (key + 7) % 12
    elif mode == 5: # Mixolydian to Major
        key = (key + 5) % 12
    elif mode == 6: # Locrian to Major
        key = (key + 1) % 12
    mode = 0
    return key, mode

def write_history(history, hist, epoch, errCntAvg):
    history[0].append(epoch)
    history[1].append(round(hist.history['loss'][0], 2))
    history[2].append(round(hist.history['val_loss'][0], 2))
    history[3].append(round(errCntAvg, 2))
    return history

def Melody_Matrix_to_Section_Composed(melody_matrix):
    section = int(np.ceil(melody_matrix.shape[0]/16.0))
    section_composed = np.zeros((section,13), dtype=np.int)
    for m in xrange(128):
        mimax = np.amax(melody_matrix[m])
        mi = np.argmax(melody_matrix[m])
        if mimax == 0:
            section_composed[m/16][0] += 1
            continue
        for mm in xrange(12):
            if mm == mi:
                section_composed[m/16][mm+1] += 1
    return section_composed

def top3notes(chord):
    idx = np.argsort(chord)
    idx[idx < 9] = 0
    idx[idx >= 12-3] = 1
    return idx

def Matrices_to_MIDI(melody_matrix, chord_matrix):
    #assert(melody_matrix.shape[0] == chord_matrix.shape[0])
    assert(melody_matrix.shape[1] == 12 and chord_matrix.shape[1] ==12)

    defaultMelOct = 5 # default melody octave
    defaultChrdOct = 3
    BPM = 160
    duration = 15.0/BPM
    m_start, c_start = 0,0
    length = melody_matrix.shape[0]
    song = pretty_midi.PrettyMIDI()
    agp_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano') # use for chords
    bap_program = pretty_midi.instrument_name_to_program('Bright Acoustic Piano') # use for melody
    melody = pretty_midi.Instrument(program= agp_program)
    chords = pretty_midi.Instrument(program= bap_program)

    for i in range(length):
        # Synthesizing melody
        m_note_nb_new = melody_matrix[i].tolist().index(1) if 1 in melody_matrix[i].tolist() else None
        if i == 0:
            m_note_nb_cur = m_note_nb_new
            m_time = 1
        elif m_note_nb_new == m_note_nb_cur:
            m_time += 1
        else:
            if m_note_nb_cur is not None:
                note = pretty_midi.Note(velocity=100, pitch=(m_note_nb_cur +12*(defaultMelOct + 1)) , start=m_start*duration, end=(m_start+m_time)*duration)
                melody.notes.append(note)
            m_start += m_time
            m_note_nb_cur = m_note_nb_new
            m_time = 1

        # Synthesizing chord
        chords_new = np.where(chord_matrix[i] == 1)[0]
        if i == 0:
            chords_cur = chords_new
            c_time = 1
        elif np.array_equal(chords_cur, chords_new):
            c_time +=1
        else:
            for n in chords_cur.tolist():
                note = pretty_midi.Note(velocity=100, pitch=(n + 12 * (defaultChrdOct + 1)),
                                        start=c_start * duration, end=(c_start + c_time) * duration)
                chords.notes.append(note)
            c_start += c_time
            chords_cur = chords_new
            c_time = 1

    # Adding notes from last iteration
    if m_note_nb_cur is not None:
        note = pretty_midi.Note(velocity=100, pitch=(m_note_nb_cur + 12 * (defaultMelOct + 1)),
                                start=m_start * duration, end=(m_start + m_time) * duration)
        melody.notes.append(note)
    for n in chords_cur.tolist():
        note = pretty_midi.Note(velocity=100, pitch=(n + 12 * (defaultChrdOct + 1)),
                                start=c_start * duration, end=(c_start + c_time) * duration)
        chords.notes.append(note)

    song.instruments.append(melody)
    song.instruments.append(chords)
    return song
