from gensim import corpora, models, similarities
import re
import sys
from sklearn.feature_extraction import stop_words

class MyCorpus:
    def __init__(self, file_name, dictionary):
        self.file_name = file_name
        self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.file_name):
            yield self.dictionary.doc2bow(line.lower().split())

def filter_bow(bow_vec):
    i=0
    while (i < len(bow_vec)): 
        word = bow_vec[i]
        if word[0] == '#':
            word = word[1:len(word)]
            bow_vec[i] = word #remove hashtags
        if re.match("^[a-zA-Z0-9_]*$", word):
            i = i+1
        else:
            del bow_vec[i] #remove words with special characters
    return bow_vec

def build_dictionary(file_name, stop_list): #stop_list is a set of strings
    dictionary = corpora.Dictionary(filter_bow(line.lower().split()) for line in open(file_name))
    stop_ids = [dictionary.token2id[stop_word] for stop_word in stop_list
                if stop_word in dictionary.token2id]
    dictionary.filter_tokens(stop_ids) # remove stop words
    dictionary.compactify() # remove gaps in id sequence after words that were removed
    return dictionary

def test_dict_corpus(file_name):
    #stoplist = set('for a of the and to in he she i we her his is are was were been'.split())
    stoplist = set(stop_words.ENGLISH_STOP_WORDS)
    dictionary = build_dictionary(file_name, stoplist)
    print(dictionary)
    print(dictionary.token2id)
    bow_corpus = MyCorpus(file_name, dictionary)
    for vector in bow_corpus:
        print(vector)

def model_lda(file_name):
    numtopic = 5
    stoplist = set('for a of the and to in he she i we they her his our their my your is are was were been u you lol'.split())
    dictionary = build_dictionary(file_name, stoplist)
    bow_corpus = MyCorpus(file_name, dictionary)
    lda_model = models.ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=numtopic)
    #file = open("lda_"+file_name, "wb")
    file = open("Data/lda_NON_HIV_tweets.txt", "wb")
    i = 0
    for i in range(numtopic):
        file.write(lda_model.print_topic(i, 10) + "\n")
    file.close()
        
if __name__ == "__main__":
    if len(sys.argv)>1:
        model_lda(sys.argv[1])
    else:
        print("Give a file name")

