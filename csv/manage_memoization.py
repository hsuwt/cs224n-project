from os import path, devnull, remove
import argparse
import csv


def empty_file(path):
    open(path, 'w').close()


def delete_files(univ, config):
    def delete(args):
        for prefix in univ:
            removeable = [prefix + '-' + suffix + '.npy' for suffix in [
                'chord', 'melody', 'sampleweight'
            ]]
            for rm in removeable:
                print "removing %s" % rm
                try:
                    remove(rm)
                except Exception:
                    print "Error removing %s" % rm

        # reset the bookkeeping file
        empty_file(config)
    return delete

if __name__ == '__main__':
    config = 'npy-exists.config'
    if path.exists(config):
        available = [row[0] for row in csv.reader(open(config))]
        print "Found files : %r" % available
    else:
        available = []

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    clean = subparsers.add_parser('clean')
    clean.set_defaults(func=delete_files(available, config))
    args = parser.parse_args()

    args.func(args)


