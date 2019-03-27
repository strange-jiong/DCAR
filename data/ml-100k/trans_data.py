
import sys
from operator import itemgetter, attrgetter


def trans(read_filename,write_filename):
    data=[]
    with open(read_filename, 'r') as f:
        for line in f:
            if line:
                lines = line[:-1].split("\t")
                user = int(lines[0])
                movie = int(lines[1])
                score = float(lines[2])
                time = int(lines[3])
                data.append((user, movie, score, time))

    data=sorted(data, key=itemgetter(0, 3))

    with open(write_filename,"w") as write_f:
        for each_line in data:
            # print(type(each_line))
            each_line=map(str,each_line)
            write_f.write("::".join(each_line)+"\n")

if __name__=="__main__":
    trans("ratings.data.old","ratings.data")
