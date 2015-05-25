from Tools2 import filter_bow
from sklearn.feature_extraction import stop_words
import csv
from FIM import FIM_keyword_network
from LDA import build_dictionary
import igraph

def buildDict(dict, filePath, maxLines):
    fin = open(filePath)
    lineCount = 0
    
    for line in fin:
        if lineCount >= maxLines:
            break
        lineVec = line.split(' ')
        lineVec = filter_bow(lineVec)
        for word in lineVec:
            if dict.has_key(word):
                dict[word] = dict[word] + 1
            else:
                dict[word] = 1
        lineCount = lineCount + 1
    fin.close

def filterDict(dict, min):
    for key in dict.keys():
        if dict[key] < min:
            del dict[key]

def writeDict(dict, fileDict):
    fout = open(fileDict, "wb")
    for key, val in dict.items():
        fout.write(str(key) + ' ' + str(val) + '\n')
    fout.close()

def buildEdgeList(dict, fileFIM, fileGraph):
    fout = open(fileGraph, "wb")
    fin = open(fileFIM)
    
    for line in fin:
        wordTuple = line.split(' ')[0].split(',')
        tupleFreq = float(line.split(' ')[1])
        prob0to1 = tupleFreq/float(dict[wordTuple[0]])
        if prob0to1 > 1.0: #because FIM yields bigger frequency sometimes...
            prob0to1 = 1.0
        prob1to0 = tupleFreq/float(dict[wordTuple[1]])
        if prob1to0 > 1.0:
            prob1to0 = 1.0
        if prob0to1 > 0.5:
            fout.write(wordTuple[0] + ' ' + wordTuple[1] + ' ' + str(prob0to1) + '\n')
        if prob1to0 > 0.5:
            fout.write(wordTuple[1] + ' ' + wordTuple[0] + ' ' + str(prob1to0) + '\n')
    fin.close()
    fout.close()

if __name__ == "__main__":
#     fileNames = ["#avengers_04302015-05102015_tweets.txt", "#ebola_05102014-05102015_tweets.txt", 
#              "#IOT_05102014-05102015_tweets.txt", "#maypac_03152015-05152015_tweets.txt"]
#     dir = "Data/VariousTopics"
    
    fileNames = ["3month-tweets.txt"]
    dir = "Data/3month"
    
    filePaths = []
    fileFIM = dir + '/' + "frequent_itemsets.txt"
    fileDict = dir + '/' + "dict.txt"
    fileGraph = dir + '/' + "edge_list.txt"
    maxLines = 60000
    
    #building dictionary
    dict = {}
    for fileName in fileNames:
        filePath = dir + '/' + fileName
        filePaths.append(filePath)
        print("Building dictionary for " + filePath + "...")
        buildDict(dict, filePath, maxLines) #read 60000 tweets per topic
    filterDict(dict, 600)
    writeDict(dict, fileDict)
    
    #run FIM with dict
    print("Running FIM...")
    FIM_keyword_network(filePaths, dict, 30, fileFIM, maxLines)
        
    #build graph
    print("Building a graph...")
    buildEdgeList(dict, fileFIM, fileGraph)
    
    print("All Done\n")

