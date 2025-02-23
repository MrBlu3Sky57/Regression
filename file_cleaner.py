""" Parse raw data CSV files into text files only containing the relevant attributes
"""

import csv
import sys

if __name__ == '__main__':
    try:
        file_name = sys.argv[1]
        cols = [int(x) for x in sys.argv[2:]]
    except:
        print("Did not specify input file... program, terminated") 
        sys.exit(1)


    with open("raw_data/" + file_name + '.csv', mode="r") as inp_f:
        csvFile = csv.reader(inp_f)
        with open("clean_data/" + file_name + '.txt', mode="w") as out_f:
            for line in csvFile:
                out_f.write(','.join([line[i] for i in cols]) + '\n')
    sys.exit(0)