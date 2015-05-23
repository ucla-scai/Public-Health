import re

def filter_bow(bow_vec):
    i=0
    while (i < len(bow_vec)):
        bow_vec[i] = bow_vec[i].lower() #make all characters lower case 
        word = bow_vec[i]
        if len(word)>0 and word[len(word)-1] == '\n':
            word = word[0:len(word)-1]
            bow_vec[i] = word #remove the new line character
        if len(word)>0 and word[0] == '#':
            word = word[1:len(word)]
            bow_vec[i] = word #remove hashtags
        if len(word)>0 and re.match("^[a-zA-Z0-9_]*$", word):
            i = i+1
        else:
            del bow_vec[i] #remove words with special characters
    return bow_vec