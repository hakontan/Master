import numpy as np

filaments = []

# Reading part of .txt file containing filament segments
with open("MD3_cat.txt.NDnet_s2.up.NDskl.a.NDskl") as infile, open("filament_segs_MD3_s2.txt", 'w') as outfile:
    copy = False
    for line in infile:
        if line.strip() == "[FILAMENTS]":
            copy = True
            continue
        elif line.strip() == "[CRITICAL POINTS DATA]":
            copy = False
            continue
        elif copy:
            outfile.write(line.lstrip())


# Storing X,Y and Z components of start- and end points for each filament segment in a given filament
with open("filament_segs_MD3_s2.txt") as infile:
    infile.readline()
    filament_nr = 0
    for line in infile:
        nr_segs = int(line.split()[2])
        segs = np.zeros(shape=(nr_segs, 3))
        for i in range(nr_segs):
            line_seg = infile.readline()
            segs[i, 0] = float(line_seg.split()[0])
            segs[i, 1] = float(line_seg.split()[1])
            segs[i, 2] = float(line_seg.split()[2])
        filaments.append(segs)
    filament_nr += 1
np.save("filament_segs/filament_segs_MD3_s2", np.array(filaments, dtype="object"))
