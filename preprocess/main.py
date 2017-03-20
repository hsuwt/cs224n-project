import os
import track

pathlist = []
pathcount = 0
availablepathindex = []
availablepathcount = 0
Directory = '/Users/ericyang/224n/project/data'
Directory = '.'
for root, dirs, files in os.walk(Directory):
    for file in files:
        if file.endswith(".mid"):
            print pathcount,
            if pathcount <= 76831:
                pathcount += 1
                print ""
                continue
            path = (os.path.join(root, file))
            localpath = path[len(Directory):]

            pathlist.append(localpath)
            pathcount += 1

            available, label = track.labelmidi(path)
            if available:
                print "V",  localpath,
                availablepathindex.append(pathcount-1)
                availablepathcount += 1

                track.labeltocsv(Directory, localpath, label)
            print ""
        #if pathcount == 3:
        #    break
print availablepathcount, "/", pathcount
