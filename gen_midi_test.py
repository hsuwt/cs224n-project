from os import listdir
from os.path import isfile, join
mypath = '../pred/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
midi_output_path = 'midi_out/'
M = np.load('csv/normal-melody.npy')
C = np.load('csv/normal-chord.npy')

for i in range(10):
    song_ground = util.matrices2midi(M[i], C[i])
    song_ground.write(midi_output_path + str(i) + '_ground.mid')
    for file in files:
        print file
        C_pred = np.load('../pred/' + file)
        song_pred = util.matrices2midi(M[i], C_pred[i])
        song_pred.write(midi_output_path + str(i) + '_' + file[:-4] + '.mid')
for f in files:
    print f
    for i in range(args.number):
        C_pred = np.load('../pred/' + f)
        song_pred = util.matrices2midi(M[i], C_pred[i])
        song_pred.write(midi_output_path + str(i) + '_' + f[:-4] + '.mid')

