import os
import numpy
from stat import S_ISREG, ST_CTIME, ST_MODE
import numpy as np
import os, sys, time

current_path = os.getcwd()
pts_path = os.path.join(current_path, 'TrainSet/all_data/')
pts_file_list = os.listdir(pts_path)

# Sort file names into creation date
data = (os.path.join(pts_path, name) for name in pts_file_list)
data = ((os.stat(path), path) for path in data)
data = ((stat[ST_CTIME], path) for stat, path in data if S_ISREG(stat[ST_MODE]))

file_name = []

for cdate, path in sorted(data):
    file_name.append(os.path.basename(path))



def read_png_path(mode):

    for file in file_name:
        with open(mode + ".txt", 'a') as f :
            f.write((file) + "\n")

def main():
    read_png_path("pts_49984")

if __name__ == '__main__':
    main()
