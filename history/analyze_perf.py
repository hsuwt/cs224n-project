import csv
from collections import defaultdict, deque

def get_last_row(csv_filename):
    """
    getting last row from csv file
    thanks to
     http://stackoverflow.com/questions/20296955/reading-last-row-from-csv-file-python-error
    """
    with open(csv_filename, 'r') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow


def extract_data(pat):
    file0 = csv.reader(open(pat.format(0.0)))
    header = file0.next()
    lines = defaultdict(list)

    for i in range(0, 11):
        ratio = i / 10.  # floating number 0.0 - 1.0
        filename = pat.format(ratio)
        print "reading file %s" % filename
        lastrow = get_last_row(filename)
        for k, v in zip(header, lastrow):
            lines[k].append(v)

    return header, lines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', nargs='?', default='GRU_LM_onehot_nodes128')
    parser.add_argument('--key', nargs='?', default=None)
    args = parser.parse_args()

    filename_pat = '/home/nykh/Documents/cs224n/proj/code/history/' + args.prefix + '_{:1}.csv'
    header, lines = extract_data(filename_pat)
    del lines['epoch']
    del header[0]

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sbn  # optional, purely for aesthetic

    xs = np.arange(0, 11).astype(np.float) / 10
    assert len(xs) == len(lines['train1'])

    if args.key is None:
        for hdr, line in lines.items():
            plt.plot(xs, line, '.-', label=hdr)
        plt.legend(loc=1)
        filename = 'perrfplot-' + args.prefix + '.png'

    else:
        key1, key12 = args.key + '1', args.key + '12'
        assert key1 in lines, 'The key specified is not legal, choose from train, val, err'
        plt.plot(xs, lines[key1], '.-', label=key1)
        plt.plot(xs, lines[key12], '.-', label=key12)
        plt.legend(loc=1)
        filename = 'perrfplot-' + args.prefix + '-' + args.key + '.png'

    plt.savefig(filename)
    print "Output written to %s" % filename
