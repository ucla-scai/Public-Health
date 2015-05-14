import sys
import csv
import re
from Tools2 import filter_bow

def parse_csv_to_txt_HIV():
    fout = open('Data/HIV_tweets.txt', 'wb')
    with open('Data/parsed_tweets.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        for row in hiv_reader:
            if row[2] == '1': #parse only HIV-related tweets
                bow = row[1].lower().split(' ')
                bow = filter_bow(bow)
                if len(bow) > 0:
                    fout.write(' '.join(bow) + '\n')
    fout.close()
    
def parse_csv_to_txt_non_HIV():
    fout = open('Data/NON_HIV_tweets.txt', 'wb')
    with open('Data/parsed_tweets.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        for row in hiv_reader:
            if row[2] != '1': #parse only HIV-related tweets
                bow = row[1].lower().split(' ')
                bow = filter_bow(bow)
                if len(bow) > 0:
                    fout.write(' '.join(bow) + '\n')
    fout.close()
                    
if __name__ == "__main__":
    #parse_csv_to_txt_HIV()
    parse_csv_to_txt_non_HIV()