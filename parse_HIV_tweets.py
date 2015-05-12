import sys
import csv

def parse_and_write(outfile_name):
    with open('Data/tweets_0001_1000.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        with open(outfile_name, 'wb') as outfile:
            hiv_writer = csv.writer(outfile, delimiter=',')
            hiv_writer.writerow(['id', 'text', 'HIV'])
            st = 0
            for row_in in hiv_reader:
                if(st==0): #ignore the first row
                    st = 1
                else:
                    row_vec = [row_in[9], row_in[28], row_in[39]] #use Sean's classification; 39th column
                    hiv_writer.writerow(row_vec)
    
    with open('Data/tweets_1001_5000.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        with open(outfile_name, 'a') as outfile:
            hiv_writer = csv.writer(outfile, delimiter=',')
            st=0
            for row_in in hiv_reader:
                if(st==0):
                    st = 1
                else:
                    if(row_in[2] != ""): #ignore rows that are not classified
                        hiv_writer.writerow(row_in)
                        
    with open('Data/tweets_5001_7000_sean.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        with open(outfile_name, 'a') as outfile:
            hiv_writer = csv.writer(outfile, delimiter=',')
            st=0
            for row_in in hiv_reader:
                if(st==0):
                    st = 1
                else:
                    row_vec = [row_in[0], row_in[1], row_in[2]] #only retrieve id, text, hiv columns
                    hiv_writer.writerow(row_vec)



if __name__ == "__main__":
    if len(sys.argv)>1:
        parse_and_write(sys.argv[1])
    else:
        print("Give a file name")