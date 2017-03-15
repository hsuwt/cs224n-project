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

    res = []
    Len = len(categories.values()[0])
    for i in range(Len):
        if not allthesame(List[i][0] for List in categories.values()):
            raise ValueError('Some file of index %s are missing!' % categories.values()[0][0][0])
        res.append({cat: val[i][1] for cat, val in categories.items()})
    return res


def span_formation(notes, mask, return_new_sw=False):
    return notes


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
        melody, chord, sample_weights = f['melody'], f['chord'], f['sw']
        m = np.load(join(path, melody))
        c = np.load(join(path, chord))
        sw = np.load(join(path, sw))
        melody2, sw2 = span_formation(m, mask=sw, return_new_sw=True)
        chord2 = span_formation(chord, mask=sw)

