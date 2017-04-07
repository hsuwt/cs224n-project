import csv
from collections import defaultdict
import numpy as np


def extract_err_timeseries(fn, with_epoch=False):
    print 'reading from %s' % fn
    with open(fn) as csvfile:
        cr = csv.reader(csvfile)
        header = cr.next()
        lines = defaultdict(list)

        for row in cr:
            for k, v in zip(header, row):
                lines[k].append(v)
        if not with_epoch:
            del lines['epoch']

        for k in lines.keys():
            lines[k] = np.array(lines[k])

        return header, lines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', nargs='?', default='GRU_LM_onehot_nodes128')
    parser.add_argument('--key', nargs='?', default=None)
    args = parser.parse_args()

    filename = '/home/nykh/Documents/cs224n/proj/code/history/' + args.prefix + '_0.5.csv'

    import matplotlib.pyplot as plt
    import seaborn as sbn  # optional, purely for aesthetic

    _, errs = extract_err_timeseries(filename, with_epoch=True)
    xs = errs['epoch']
    del errs['epoch']

    if args.key is None:
        for hdr, line in errs.items():
            plt.plot(xs, line, '.-', label=hdr)
        plt.legend(loc=1)
        output_filename = 'errplot-' + args.prefix + '.png'

    else:
        keys = args.key.split()
        assert all(key in errs for key in keys), 'The key specified is not legal'
        for key in keys:
            plt.plot(xs, errs[key], '.-', label=key)
        plt.legend(loc=1)
        output_filename = 'errplot-' + args.prefix + '-' + '_'.join(keys) + '.png'

    plt.savefig(output_filename)
    print "Output written to %s" % output_filename
