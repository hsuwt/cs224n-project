import sys
import os
import numpy as np
import pretty_midi
import csv


def rotate(Chroma, semitone):
    if semitone == 0:
        return Chroma
    return np.concatenate((Chroma[-semitone:], Chroma[:Chroma.shape[0] - semitone]), axis=0)

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


def chroma2chord_v2(Chroma):
    # v2 is way faster
    notes = np.where(Chroma)[0]
    interval = np.diff(notes)
    if len(notes) == 3:
        if all(interval == np.array([4, 3])): return [notes[0], 1]  # maj root position
        if all(interval == np.array([3, 5])): return [notes[2], 1]  # maj 1st inversion
        if all(interval == np.array([5, 4])): return [notes[1], 1]  # maj 2nd inversion
        if all(interval == np.array([3, 4])): return [notes[0], 2]  # maj root position
        if all(interval == np.array([4, 5])): return [notes[2], 2]  # maj 1st inversion
        if all(interval == np.array([5, 3])): return [notes[1], 2]  # maj 2nd inversion
        return [0, 6]
    if len(notes) == 4:
        if all(interval == np.array([4, 3, 4])): return [notes[0], 3]  # maj7 root position
        if all(interval == np.array([3, 4, 1])): return [notes[3], 3]  # maj7 1st inversion
        if all(interval == np.array([4, 1, 4])): return [notes[2], 3]  # maj7 2nd inversion
        if all(interval == np.array([1, 4, 3])): return [notes[1], 3]  # maj7 3rd inversion
        if all(interval == np.array([4, 3, 3])): return [notes[0], 4]  # dmn7 root position
        if all(interval == np.array([3, 3, 2])): return [notes[3], 4]  # dmn7 1st inversion
        if all(interval == np.array([3, 2, 4])): return [notes[2], 4]  # dmn7 2nd inversion
        if all(interval == np.array([2, 4, 3])): return [notes[1], 4]  # dmn7 3rd inversion
        if all(interval == np.array([3, 4, 3])): return [notes[0], 5]  # min7 root position
        if all(interval == np.array([4, 3, 2])): return [notes[3], 5]  # min7 1st inversion
        if all(interval == np.array([3, 2, 3])): return [notes[2], 5]  # min7 2nd inversion
        if all(interval == np.array([2, 3, 4])): return [notes[1], 5]  # min7 3rd inversion
        return [0, 6]
    if len(notes) == 2:
        if all(interval == np.array([7])): return [notes[0], 7]  # power chord
        if all(interval == np.array([5])): return [notes[1], 7]  # power chord
        return [0, 6]
    return [0, 6]


def chroma2chord_v1(chroma):
    for i in range(12):
        chroma_shifted = rotate(chroma, i)
        if all(chroma_shifted == np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])): return [(12 - i) % 12, 1]  # maj
        if all(chroma_shifted == np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])): return [(12 - i) % 12, 2]  # min
        if all(chroma_shifted == np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])): return [(12 - i) % 12, 3]  # maj7
        if all(chroma_shifted == np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])): return [(12 - i) % 12, 4]  # 7
        if all(chroma_shifted == np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])): return [(12 - i) % 12, 5]  # min7
    return [0, 0]


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

_qualifier = {
    1: 'Maj', 2: 'Min', 3: 'Maj7', 4: '7', 5: 'Min7'
}


def printChord(Chroma):
    r, t = chroma2chord_v2(Chroma)
    S = notes._list[r] + _qualifier[t] if t != 6 else 'N'
    print '%s\t' % S


def printChordProgression(y, cp):
    assert y.shape == cp.shape
    assert y.shape[1] == 128
    assert y.shape[2] == 12

    n_song = len(cp)
    for i in range(n_song):
        print 'song %d\t answ:' % i
        for j in range(8):
            printChord(y[i][16 * j])
        print '\nsong %d\t pred:' % i
        for j in range(8):
            printChord(cp[i][16 * j])
        print '\nsong %d\t diff:' % i
        cnt = 0
        for j in range(8):
            tmp = np.sum(abs(y[i][16 * j] - cp[i][16 * j]))
            cnt += tmp
            print '%d\t' % tmp
        print '%d\n' % cnt


def pred2chord(pred):
    pred = pred.reshape((pred.shape[0], 12, 8, pred.shape[1] / 12 / 8))
    pred = np.mean(pred, axis=3)
    for i in range(len(pred)):
        for j in range(len(pred[0][0])):
            pred[i][:][:, j] = closestChord(pred[i][:][:, j])


def toCandidate(CP, allCP, bestN, criteria):
    ret = np.zeros_like(CP)
    bestIdx = np.zeros((len(CP), bestN), dtype=int)
    for i in range(len(CP)):
        minDis = sys.maxint
        minIdx = 0
        dis = np.zeros((len(allCP)))
        for j in range(len(allCP)):
            if criteria == 'L1':
                dis[j] = np.sum(abs(CP[i] - allCP[j]))
            elif criteria == 'L2':
                dis[j] = np.sqrt(np.sum(np.square(CP[i] - allCP[j])))
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
            dis[j] = np.sum(abs(CP[i] - allCP[j]))
        bestIdx[i] = np.argsort(dis)[:bestN]
    return bestIdx


def csv2npy():
    max_length = 1024
    for j in range(4, 7):
        nb_train = sum([len(files) for r, d, files in os.walk("../dataset_bk/melody/part" + str(j))])
        C = np.zeros((nb_train, max_length, 12))
        M = np.zeros((nb_train, max_length, 12))
        sample_weight = np.zeros((nb_train, max_length))
        train_idx = 0
        for root, _, files in os.walk("../dataset_bk/melody/part" + str(j)):
            for m_file_name in files:
                m_file_name_path = os.path.join(root, m_file_name)
                c_file_name_path = m_file_name_path.replace("/melody/", "/chord/", 1)
                m_file_matrix = np.genfromtxt(m_file_name_path, delimiter=',')
                c_file_matrix = np.genfromtxt(c_file_name_path, delimiter=',')
                if len(m_file_matrix) == 0 or len(c_file_matrix) == 0: continue
                m_file_matrix = m_file_matrix[:, :max_length]
                c_file_matrix = c_file_matrix[:, :max_length]
                seq_len = min(m_file_matrix.shape[1], c_file_matrix.shape[1])
                tmp_m, tmp_c = np.zeros((12, seq_len)), np.zeros((12, seq_len))
                for i in range(128 / 12):
                    tmp_m = np.logical_or(tmp_m, m_file_matrix[i * 12:(i + 1) * 12, :seq_len])
                    tmp_c = np.logical_or(tmp_c, c_file_matrix[i * 12:(i + 1) * 12, :seq_len])
                sample_weight[train_idx, :seq_len] = np.ones((1, seq_len))
                M[train_idx, :seq_len], C[train_idx, :seq_len] = np.swapaxes(tmp_m, 0, 1), np.swapaxes(tmp_c, 0, 1)
                train_idx += 1
        C = C[:train_idx]
        M = M[:train_idx]
        sample_weight = sample_weight[:train_idx]
        print C.shape
        print M.shape
        print sample_weight.shape
        np.save('~/npy/chord_csv' + str(j) + '.npy', C.astype(int))
        np.save('~/npy/melody_csv' + str(j) + '.npy', M.astype(int))
        np.save('~/npy/sw_csv' + str(j) + '.npy', sample_weight.astype(int))
        print("saving csv" + str(j) + ".npy")


def print_result(pred, y, Y, alg, bestN, printCP=False, verbose=False):
    nb_test = pred.shape[0]
    if 'L2' in alg.model:
        pred, bestNIdx = toCandidate(pred, Y, bestN, 'L2')
        norm = np.sqrt(np.sum(np.square(pred - y))) / 128.0 / nb_test
    else:
        pred, bestNIdx = toCandidate(pred, Y, bestN, 'L1')
        norm = np.sum(abs(pred - y)) / 128.0 / nb_test
    numUniqIdx = len(np.unique(bestNIdx))
    if printCP:
        printChordProgression(y, pred)
    if verbose:
        print('num of unique idx  = %d/%d' % (numUniqIdx, nb_test))
        print('norm after mapping = %.3f' % (norm))
    return bestNIdx, numUniqIdx, norm


class LUT(object):
    def __init__(self, name, _list):
        self.__name__ = name
        self._list = list
        assert len(_list) > 0
        self._dict = {x: i for i, x in enumerate(_list)}

    def lookup(self, i):
        try:
            return self._list[i]
        except:
            print 'no match in {}: {}'.format(self.__name__, i)
            return self._list[0]

    def lookup_idx(self, x):
        return self._dict.get(x, 0)

    def __len__(self):
        return len(self._list)


notes = LUT('notes', ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
modes = LUT('modes', ['Major', 'Minor', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Locryan'])
types = LUT('types', ['Non-chord', 'Maj', 'Min', 'Maj7', 'Dominant7', 'Min7', 'Complex'])


def toMajKey(key, mode):
    if mode == 1:  # Minor to Major
        key = (key + 3) % 12
    elif mode == 2:  # Dorian to Major
        key = (key + 10) % 12
    elif mode == 3:  # Phrygian to Major
        key = (key + 8) % 12
    elif mode == 4:  # Lydian to Major
        key = (key + 7) % 12
    elif mode == 5:  # Mixolydian to Major
        key = (key + 5) % 12
    elif mode == 6:  # Locrian to Major
        key = (key + 1) % 12
    mode = 0
    return key, mode


def melody_matrix_to_section_composed(melody_matrix):
    section = int(np.ceil(melody_matrix.shape[0] / 16.0))
    section_composed = np.zeros((section, 13), dtype=np.int)
    for m in xrange(128):
        mimax = np.amax(melody_matrix[m])
        mi = np.argmax(melody_matrix[m])
        if mimax == 0:
            section_composed[m / 16][0] += 1
            continue
        for mm in xrange(12):
            if mm == mi:
                section_composed[m / 16][mm + 1] += 1
    return section_composed


def top3notes(chord):
    idx = np.argsort(chord)
    idx[idx < 9] = 0
    idx[idx >= 12 - 3] = 1
    return idx


def matrices2midi(melody_matrix, chord_matrix):
    # assert(melody_matrix.shape[0] == chord_matrix.shape[0])
    assert (melody_matrix.shape[1] == 12 and chord_matrix.shape[1] == 12)

    defaultMelOct = 5  # default melody octave
    defaultChrdOct = 3
    BPM = 160
    duration = 15.0 / BPM
    m_start, c_start = 0, 0
    length = melody_matrix.shape[0]
    song = pretty_midi.PrettyMIDI()
    agp_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')  # use for chords
    bap_program = pretty_midi.instrument_name_to_program('Bright Acoustic Piano')  # use for melody
    melody = pretty_midi.Instrument(program=agp_program)
    chords = pretty_midi.Instrument(program=bap_program)

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
                note = pretty_midi.Note(velocity=100, pitch=(m_note_nb_cur + 12 * (defaultMelOct + 1)),
                                        start=m_start * duration, end=(m_start + m_time) * duration)
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
            c_time += 1
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


def smooth(C):
    ydim = C.shape[2]
    ret = C.reshape((-1, 128 / 8, 8, ydim))
    ret = np.average(ret, axis=2)
    return np.repeat(ret, 8, axis=1)

def rotateNotes(C, semitone):
    if semitone == 0:
        return C
    return np.concatenate((C[:,:,-semitone:], C[:,:,:12-semitone]), axis=2)

def dataAug(C):
    newC = C+0
    for i in range(11):
        newC = np.concatenate((newC, rotateNotes(C, i+1)), axis=0)
    return newC
