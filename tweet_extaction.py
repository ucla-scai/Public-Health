from Tools2 import filter_bow
import numpy as np
import json
import re
from datetime import datetime

def split_word(tweets):
    tweet_words = []
    for tweet in tweets:
        words = re.findall(r'[#]+', tweet)
        words_filtered = [word.lower() for word in words if len(word) >= 3]
        tweet_words.append(' '.join(words_filtered))
    tweet_words.pop(0)
    return tweet_words

def tweets_lang_filter(path):
    tweet_vec = []
    lang_count = 0
    fout = open("Data/3month/3month_tweets_parsed.txt", "wb")
    with open(path, 'rU') as json_data:
        for json_line in json_data:
            tweet = json.loads(json_line)
            if not tweet["tweet"].has_key("lang"):
                continue
            if tweet["tweet"]["lang"] == "en":
                lineVec = tweet["tweet"]["text"].split(' ')
                filter_bow(lineVec)
                fout.write(' '.join(lineVec) + '\n')
#                 fout.write(tweet["tweet"]["text"] + '\n')
#                 print(tweet["tweet"]["text"])
    fout.close()
    
#                 timestamp = tweet["firstpost_date"]
#                 timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S ')
#                 text = tweet["tweet"]["text"]
#                 text = re.findall(r'#\w+', text)
#                 if len(text) > 0:
#                     tweet_vec.append(timestamp + ' '.join(text))
#             else:
#                 lang_count += 1
#     np.savetxt('tweets_hashtag.txt', tweet_vec, fmt='%s') 
#     print 'tweets in English %d, skip tweets %.2f%% (total %d)' \
#           % (len(tweet_vec), lang_count * 100.0/(lang_count + len(tweet_vec)), lang_count + len(tweet_vec))
#     return tweet_vec

if __name__ == '__main__':
    file_path = 'Data/3month/3month_tweets.txt'
    tweets_lang_filter(file_path)
    