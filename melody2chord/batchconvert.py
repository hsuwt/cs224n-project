import os
import sys

files = sorted([f for f in os.listdir('.') if f.endswith('mid')])

for f in files:
    print 'converting ' + f
    fn = f[:-4]
    print fn
    cmd = 'timidity {0}.mid -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {0}.mp3'.format(fn)
    os.system(cmd)
