"""
This is a preprocessing step that transform already parsed Numpy array into span/interval format

This relies on other preprocessing code to parse MIDI into Numpy array that represent the pitch of
each 16th note. The new representation ('span representation') merges adjacent note and add a new
entry to the vector which represent the number of notes merged this way.

The program goes through each Numpy array file once.
"""
import re
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def allthesame(Iter):
    i = Iter.next()
    return all(x == i for x in Iter)


def strip_dirs(fs):
    """
    strip the dir part from a list of files
    """
    if '/' in fs[0]:
        dir = fs[0].rsplit('/', 1)[0]
        return [f.rsplit('/', 1)[-1] for f in fs], dir
    else:
        return fs


def categorize(fs):
    fs = sorted(fs)
    categories = defaultdict(list)
    findnumber = re.compile('\d+')
    for f in fs:
        cat = f.split('_')[0]
        mat = findnumber.search(f)
        if mat is not None:
            Id = (mat.group(0))
        else:
            Id = 0
        categories[cat].append((Id, f))

    results = []
    Len = len(categories.values()[0])
    for i in range(Len):
        Id = int(categories.values()[0][0][0])
        if not allthesame(List[i][0] for List in categories.values()):
            raise ValueError('Some file of index %d are missing!' % Id)
        res = {cat: val[i][1] for cat, val in categories.items()}
        res['index'] = Id
        results.append(res)
    return results


def span_formation(notes, mask, return_mask=False):
    """
    span_formation goes through notes once, merging adjacent notes and indicate merging in the corresponding
    length column. It also keep a running sum, and computes a new mask

    :param notes:
    :param mask: the old mask
    :param return_mask: whether to return mask
    :return: (new notes, new mask)
    """
    mask = mask.astype(bool)

    num_songs, Len = notes.shape[0], notes.shape[1]
    new_notes = np.zeros([num_songs, Len, 13])
    new_mask = np.zeros([num_songs, Len])
    changes = (notes[:, :-1] - notes[:, 1:]).sum(axis=2)
    print "song[10]= %r\n" % notes[10]
    empty_note = np.zeros([1, 13])
    with tqdm(total=num_songs) as pbar:
        for i, diffs in enumerate(changes):
            song = notes[i, mask[i]]

            ends = diffs.nonzero()[0]
            if len(ends) == 0:
                new_song = np.zeros([L, 13])
                new_song[0, -1] = Len
                new_mask = np.zeros(Len)

            if ends[-1] == len(song) - 1:
                # avoid error when song is shortened than Len by mask
                ends = ends[:-1]
            # each index in ends is a end position for a note,
            # because the note is different from the next
            starts = np.append(0, ends + 1)
            ends = np.append(ends, Len-1)
            # therefore the next positions are the start pos, plus, of course, 0
            lens = ends - starts + 1
            try:
                new_song = np.concatenate([song[starts], np.matrix(lens).T], axis=1)
            except IndexError as ex:
                print "i=", i
                raise ex
            totalLen = lens.sum()
            if totalLen < Len:
                # We would like the total length to be
                new_song = np.concatenate(new_song, empty_note)
                new_song[-1, -1] = Len - totalLen
            num_notes = new_song.shape[0]
            new_notes[i, :num_notes] = new_song
            if return_mask:
                new_mask[i, :num_notes] = 1
            pbar.update(1)

    if return_mask:
        return new_notes, new_mask
    else:
        return new_notes


def save(ar, basefilename):
    np.save(ar, basefilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform Numpy array of 16th notes into span format')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    files = args.files
    files, dir = strip_dirs(files)
    files = categorize(files)

    path = dir + '/'
    def join(path, file):
        return path + file

    for f in files:
        print "Converting file set %d" % f['index']
        melody, chord, sample_weights = f['melody'], f['chord'], f['sw']
        print "Loading files..."
        m = np.load(path + '/' + melody)
        c = np.load(path + '/' + chord)
        sw = np.load(path + '/' + sample_weights)
        print "Converting melody..."
        print "Converting chords\n"
        melody2, sw2 = span_formation(m, mask=sw, return_mask=True)
        chord2 = span_formation(c, mask=sw)

        save(melody2, '{}/melody_span{}.npy'.format(path, i))
        save(chord2, '{}/chord_span{}.npy'.format(path, i))
        save(sw2, '{}/sw_span{}.npy'.format(path, i))
