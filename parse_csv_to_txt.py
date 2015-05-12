import sys
import csv
import re

def filter_tweet_bow(bow_vec):
    i=0
    while (i < len(bow_vec)): 
        word = bow_vec[i]
        if re.match("^[a-zA-Z0-9_]*$", word) and len(word)>0:
            i = i+1
        else:
            del bow_vec[i] #remove words with special characters
    return bow_vec

def parse_csv_to_txt():
    fout = open('Data/HIV_tweets.txt', 'wb')
    with open('Data/parsed_tweets.csv', 'rU') as infile:
        hiv_reader = csv.reader(infile, delimiter=',')
        for row in hiv_reader:
            bow = row[1].lower().split(' ')
            bow = filter_tweet_bow(bow)
            print(bow)
            fout.write(' '.join(bow) + '\n')
    fout.close()
                    
if __name__ == "__main__":
    parse_csv_to_txt()