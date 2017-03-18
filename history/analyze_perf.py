import csv
from collections import defaultdict, deque

filename_pat = '/home/nykh/Documents/cs224n/proj/code/history/GRU_LM_onehot_nodes128_{:1}.csv'


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
    ratio = 0.0
    file0 = csv.reader(open(filename_pat.format(ratio)))
    header = file0.next()
    lines = defaultdict(list)

    for i in range(0, 11):
        ratio = i / 10.  # floating number 0.0 - 1.0
        filename = filename_pat.format(ratio)
        print "reading file %s" % filename
        lastrow = get_last_row(filename)
        for k, v in zip(header, lastrow):
            lines[k].append(v)

    return header, lines