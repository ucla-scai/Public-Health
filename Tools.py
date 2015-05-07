'''
Created on Feb 26, 2015
@author: Wenchao
'''
# -*- coding: UTF-8 -*-
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import csv
import numpy as np
from string import punctuation
import re
from sklearn import cross_validation
import json

def filter_tweets(path):
    with open(path, 'rU') as f:
        reader = csv.reader(f)
        data = list(reader)
        mat_data = np.array(data)
        mat_data = mat_data.transpose()
        labels = mat_data[2]
        texts = mat_data[1]
    valid_idx = [idx for idx, label in enumerate(labels) if label == '0' or label =='1']
    labels = labels[valid_idx].astype(np.int)
    texts = texts[valid_idx]
    return texts, labels

def filter_tweets_first(path):
    ''' filtering the first 1000 tweets with different format
    '''
    with open(path, 'rU') as f:
        reader = csv.reader(f)
        data = list(reader)
        mat_data = np.array(data)
        mat_data = mat_data.transpose()
        labels = mat_data[39:43]    # professor label, reason, student1 label, student2 label
        texts = mat_data[28]
    labels = labels[0]              # only use professor label
    valid_idx = [idx for idx, label in enumerate(labels) if label == '0' or label =='1']
    labels = labels[valid_idx].astype(np.int)
    texts = texts[valid_idx]
    # vote for labels
    #labels = labels.sum(axis=0)
    #labels[labels <= 1] = 0
    #labels[labels >= 2] = 1
    return texts, labels
    
def split_tweets(tweets):
    tweet_words = []
    features = []
    for tweet in tweets:
        ext_feature = []
        if tweet.find('@') != -1:
            ext_feature.append(1)
        else:
            ext_feature.append(0)
            
        if tweet.find('http:') != -1:
            ext_feature.append(1)
        else:
            ext_feature.append(0)   
         
        if tweet.find('!!!') != -1 or tweet.find('???') != -1:
            ext_feature.append(1)
        else:
            ext_feature.append(0)    
            
        if tweet.count('#') >= 2:
            ext_feature.append(1)
        else:
            ext_feature.append(0)   
            
        if tweet.find(':)')!= -1 or tweet.find(':-)')!= -1 or tweet.find(':]')!= -1 or tweet.find('=]')!= -1:
            ext_feature.append(1)
        else:
            ext_feature.append(0)   
        
        if tweet.find(':(')!= -1 or tweet.find(':-(')!= -1 or tweet.find(':[')!= -1 or tweet.find('=[')!= -1:
            ext_feature.append(1)
        else:
            ext_feature.append(0)
        features.append(ext_feature)
        words = re.findall(r'[\w#@]+', tweet)
        words_filtered = [word.lower() for word in words if len(word) >= 3]
        tweet_words.append(' '.join(words_filtered))
    return tweet_words, features
    #TODO: Link remove
    
def load_data():
    '''
    process all three dataset provided by Sean
    @return: all_tweets_word, labels, data_idx
    '''
    data1 = './Data/tweets_0001_1000.csv'       # First ~1000 tweets with label 
    data2 = './Data/tweets_1001_5000.csv'       # Second ~4000 tweets with label
    data3 = './Data/tweets_5001_7000_sean.csv'  # Third 2000 with label + confidence, revised by Sean
    #data3 = './Data/tweets_5001_7000.csv'
    
    tweets1, labels1 = filter_tweets_first(data1)
    tweets2, labels2 = filter_tweets(data2)
    tweets3, labels3 = filter_tweets(data3)

    tweets_word1 = split_tweets(tweets1)[0]
    tweets_word2 = split_tweets(tweets2)[0]
    tweets_word3 = split_tweets(tweets3)[0]
    
    all_tweets_word = []
    all_tweets_word.extend(tweets_word1)
    all_tweets_word.extend(tweets_word2)
    all_tweets_word.extend(tweets_word3)
    labels = list(labels1) + list(labels2) + list(labels3)
    labels = np.array(labels)
    data_idx = [len(labels1), len(labels2), len(labels3), len(labels)]
    return all_tweets_word, labels, data_idx

def data_desc():
    print '------ data description ------'
    print 'data1: tweets from 1-1000, 996 with labels'
    print 'data2: tweets from 1000-5000, 3394 with labels'
    print 'data3: tweets from 5000-7000, revised by Sean, 1997 with labels'
    print 'data4: data1 + data2'
    print 'data5: data1 + data3'
    print 'data6: data2 + data3'
    print '-'*30
    
def tweets_stream(path):
    tweet_vec = []
    with open(path, 'rU') as json_data:
        for json_line in json_data:
            tweet = json.loads(json_line)
            text = tweet["tweet"]["text"]
            tweet_vec.append(text)
    tweet_vec = np.array(tweet_vec)
    return split_tweets(tweet_vec)

def split_word(tweets):
    tweet_words = []
    for tweet in tweets:
        words = re.findall(r'[\w#@]+', tweet)
        words_filtered = [word.lower() for word in words if len(word) >= 3]
        tweet_words.append(' '.join(words_filtered))
    tweet_words.pop(0)
    return tweet_words
    #TODO: Link remove

# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test, feature_names):
    #print('_' * 80)
    #print("Training: ")
    #print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    #print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    #print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    print("five-fold cross validation:")
    #scores = cross_validation.cross_val_score(clf, X, labels, cv=5)
    #print scores

    if hasattr(clf, 'coef_'):
        #print("dimensionality: %d" % clf.coef_.shape[1])
        #print("density: %f" % density(clf.coef_))
        print("top 10 keywords per class:")
        top10 = np.argsort(clf.coef_)[0, -10:]
        print("HIV: %s" % (" ".join(map(str, feature_names[top10]))))
        tail10 = np.argsort(clf.coef_)[0, :10]
        print("NO HIV: %s" % (" ".join(map(str,feature_names[tail10]))))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def benchmark_cv(clf, X, labels):
    scores = cross_validation.cross_val_score(clf, X, labels, cv=5)
    return scores

if __name__ == '__main__':
    pass
#     load_data()
#     
#     path = 'hiv_tweets.csv'
#     tweets, labels = to_list(path)
#     #print tweets, labels
#     
#     vectorizer = CountVectorizer(min_df=1)
#     tweets_word = split_word(tweets)
#     #print tweets_word
#     X = vectorizer.fit_transform(tweets_word)
#     feature_names = vectorizer.get_feature_names()
#     feature_names = np.asarray(feature_names)
#     #print vectorizer.vocabulary_
#     #print X.toarray()[2]
#     labels = tweets_label(labels)
#     transformer = TfidfTransformer()
#     tfidf = transformer.fit_transform(X)
#     #X = tfidf.toarray()
#     X = X.toarray()
#     idx = len(X) * 3/4
#     X_train = X[:idx]
#     X_test = X[idx:]
#     y_train = labels[:idx]
#     y_test = labels[idx:]
#     
#     #print len(labels)
#     #print sum(labels)
#     #sys.exit('end test')
#     
#     results = []
#     for clf, name in (
#             (LogisticRegression(), "Logistic Regression"), 
#             (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#             (Perceptron(n_iter=50), "Perceptron"),
#             (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")):
#             #(KNeighborsClassifier(n_neighbors=10), "kNN"),
#             #(RandomForestClassifier(n_estimators=100), "Random forest")):
#         print('=' * 80)
#         print(name)
#         results.append(benchmark(clf))
#     i=0
#     for penalty in ["l2", "l1"]:
#         i += 1
#         print('=' * 80)
#         #print("%s penalty" % penalty.upper())
#         # Train Liblinear model
#         print 'LinearSVC_%d' % i
#         results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                                 dual=False, tol=1e-3)))
#     
#         # Train SGD model
#         print('=' * 80)
#         print 'SGDClassifier_%d' % i
#         results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                                penalty=penalty)))
#     
#     # Train SGD with Elastic Net penalty
#     print('=' * 80)
#     #print("Elastic-Net penalty")
#     print 'SGDClassifier_3'
#     results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                            penalty="elasticnet")))
#     
    ## Train NearestCentroid without threshold
    #print('=' * 80)
    #print("NearestCentroid (aka Rocchio classifier)")
    #results.append(benchmark(NearestCentroid()))
    #
    ## Train sparse Naive Bayes classifiers
    #print('=' * 80)
    #print("Naive Bayes")
    #results.append(benchmark(MultinomialNB(alpha=.01)))
    #results.append(benchmark(BernoulliNB(alpha=.01)))
    #
    #
    #class L1LinearSVC(LinearSVC):
    #
    #    def fit(self, X, y):
    #        # The smaller C, the stronger the regularization.
    #        # The more regularization, the more sparsity.
    #        self.transformer_ = LinearSVC(penalty="l1",
    #                                      dual=False, tol=1e-3)
    #        X = self.transformer_.fit_transform(X, y)
    #        return LinearSVC.fit(self, X, y)
    #
    #    def predict(self, X):
    #        X = self.transformer_.transform(X)
    #        return LinearSVC.predict(self, X)
    #
    #print('=' * 80)
    #print("LinearSVC with L1-based feature selection")
    #results.append(benchmark(L1LinearSVC()))
    #
    #
    #
