
from util import *
import numpy as np

if __name__ == "__main__":
    #saveStep()
    idx = np.genfromtxt('csv/id_template.csv', delimiter=',')

    cp = np.genfromtxt('csv/chord.csv', delimiter=',')
    cp = cp.reshape((cp.shape[0], 12, 128))
    cp = np.swapaxes(cp, 1, 2)
    root  = np.genfromtxt('csv/root.csv', delimiter=',')
    r = np.zeros((root.shape[0], 12, root.shape[1]))
    for i in range(root.shape[0]):
        for j in range(root.shape[1]):
            r[i][root[i][j]][j] = 1
    r = r.reshape((r.shape[0], r.shape[1] * r.shape[2]))

    # prediction by template
    n_song = len(idx)
    dis = np.zeros((n_song, 128))
    for i in range(n_song):
        for j in range(128):
            dis[i][j] = TPS(cp[i][j], cp[int(idx[i])][j], r[i][j], r[int(idx[i])][j], 0)
    print "template cp:",
    print np.mean(dis)

    # all Cmaj cp
    dis = np.zeros((n_song, 128))
    for i in range(n_song):
        for j in range(128):
            dis[i][j] = TPS(cp[i][j], cp[186][j], r[i][j], r[186][j], 0)
    print "all Cmaj cp:",
    print np.mean(dis)

    # random cp
    dis = np.zeros((n_song, 128))
    for i in range(n_song):
        for j in range(128):
            dis[i][j] += TPS(cp[i][j], cp[(i+100) % n_song][j], r[i][j], r[(i+100) % n_song][j], 0)
    print "random cp",
    print np.mean(dis)

