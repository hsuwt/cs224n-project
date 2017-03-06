import numpy as np
from util import *

"""
There are 2% of chord progression that are empty!!!
"""

M,m,C,c,R,r = load_data(100)
cnt = 0
cntErr = 0
cntErrEmpty = 0
badId = np.zeros((len(R), 1))
for i in range(len(R)):
    for j in range(len(R[i])):
        for k in range(len(R[i][j])):
            if R[i][j][k] and not C[i][j][k]:
                badId[i] = 1
                if np.sum(C[i][j]):
                    #print R[i][j],
                    #print C[i][j]
                    pass
                else:
                    cntErrEmpty += 1
                cntErr += 1
            cnt += 1
    if (badId[i]):
        print("%d " %(i)),



print(cnt)
print(cntErr)
print(cntErrEmpty)
print(cntErr*100/cnt)
print(cntErrEmpty*100/cnt)

