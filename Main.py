'''
Created on May 6, 2015
@author: Wenchao Yu
@contact: yuwenchao@ucla.edu
'''
# -*- coding: UTF-8 -*-
import operator
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from Tools import tweets_stream
from Tools import data_desc, load_data

def word_freq(tweets):
    word_dict = {}
    for i in range(len(tweets)):
        for j in tweets[i].split():
            if j in word_dict:
                word_dict[j] += 1
            else:
                word_dict[j] = 1
    sorted_x = sorted(word_dict.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    np.savetxt('word_frequent.txt', sorted_x, delimiter='\t', fmt='%s') 
    return sorted_x[:10]

def to_feature(tweets, tweets_new):
    # use 2-gram or remove hashtag and @username
    #vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z]+\b',min_df=1)
    #vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(tweets)
    X_new = vectorizer.transform(tweets_new)
    X = X.toarray()
    X_new = X_new.toarray()
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)
    return X, X_new, feature_names

def tweets_len_stat(all_tweets, labels):
    result = []
    result_hiv = []
    for i in range(len(labels)):
        for j in all_tweets[i].split():
            result.append(len(j));
            if labels[i] == 1:
                result_hiv.append(len(j));
    np.savetxt('tweets_len.txt', result, delimiter='\n', fmt='%u') 
    np.savetxt('tweets_hiv_len.txt', result_hiv, delimiter='\n', fmt='%u') 
    print len(labels)

def LR(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(penalty='l1')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print 'Test accuracy with Logistic Regression (l1 norm): %f' % metrics.accuracy_score(y_test, pred)
    return metrics.accuracy_score(y_test, pred), clf

def online_pred(clf, X_new):
    pred = clf.predict(X_new)
    #pred_proba = clf.predict_proba(X_new)
    counts = 0
    for i, label in enumerate(pred):
    #for label in pred:
        if label != 1 and label != 0:
            Warning('Wrong label')
        if label == 1:
            counts += 1
            #print tweets_new[i].encode('utf-8')
    print 'Online prediction: %d total tweets, %d HIV related tweets (%.2f%%).' % (len(X_new), counts, counts*100.0 / len(X_new))

def clf_test(i, j):
    data1_idx = np.arange(data_idx[0])
    data2_idx = np.arange(data_idx[0], data_idx[0] + data_idx[1])
    data3_idx = np.arange(data_idx[0] + data_idx[1], data_idx[0] + data_idx[1] + data_idx[2])
    data4_idx = np.concatenate([data1_idx, data2_idx])
    data5_idx = np.concatenate([data1_idx, data3_idx])
    data6_idx = np.concatenate([data2_idx, data3_idx])
    
    idx = [data1_idx,data2_idx,data3_idx,data4_idx,data5_idx,data6_idx]
    train_idx = idx[i-1]
    test_idx = idx[j-1]
    print 'Training with data%d, test with data%d:' % (i, j)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    clf = LR(X_train, X_test, y_train, y_test)[1]
#     if hasattr(clf, 'coef_'):
#         #print("dimensionality: %d" % clf.coef_.shape[1])
#         #print("density: %f" % density(clf.coef_))
#         print("top 10 keywords per class:")
#         top10 = np.argsort(clf.coef_)[0, -10:]
#         print("\tHIV: %s" % (" ".join(map(str, feature_names[top10]))))
#         tail10 = np.argsort(clf.coef_)[0, :10]
#         print("\tNO HIV: %s" % (" ".join(map(str,feature_names[tail10]))))

    online_pred(clf, X_new)
    print '-'*30
    
if __name__ == '__main__':
    # labeled data process
    all_tweets, labels, data_idx = load_data()
    data_desc()
    
    # online data process
    online_tweets = './Data/tweets_online.txt'         
    tweets_new = tweets_stream(online_tweets)[0]
        
    #X, X_new = to_feature(all_tweets, tweets_new)[:2]
    X, X_new, feature_names = to_feature(all_tweets, tweets_new)
    print 'Dictionary size: %d' % X.shape[1]
    print '-'*30
    
    y = labels
    y_new = np.array([0]*len(X_new))
    
    print 'Training with data2 + 1/10 online data, test with data1 + 9/10 online data'
    online_percent = len(X_new)/10
    X_train, X_test = np.concatenate([X[data_idx[0]:data_idx[0]+data_idx[1]], X_new[:online_percent]]), X[:data_idx[0]]
    y_train, y_test = np.concatenate([y[data_idx[0]:data_idx[0]+data_idx[1]], y_new[:online_percent]]), y[:data_idx[0]]
    clf = LR(X_train, X_test, y_train, y_test)[1]
    online_pred(clf, X_new)
    print '-'*30
    
    print 'Training with hiv data + 1/10 online data, test with 9/10 online data'
    online_percent = len(X_new)/10
    X_train, X_test = np.concatenate([X, X_new[:online_percent]]), X_new[online_percent:]
    y_train, y_test = np.concatenate([y, y_new[:online_percent]]), y_new[online_percent:]
    clf = LR(X_train, X_test, y_train, y_test)[1]
    if hasattr(clf, 'coef_'):
        #print("dimensionality: %d" % clf.coef_.shape[1])
        #print("density: %f" % density(clf.coef_))
        print("top 10 keywords per class:")
        top10 = np.argsort(clf.coef_)[0, -10:]
        print("\tHIV: %s" % (" ".join(map(str, feature_names[top10]))))
        tail10 = np.argsort(clf.coef_)[0, :10]
        print("\tNO HIV: %s" % (" ".join(map(str,feature_names[tail10]))))
        coef_idx = np.argsort(clf.coef_)[0,::-1]
        #np.savetxt('feature weight.txt', feature_names[coef_idx], delimiter='\n', fmt='%s') 
        np.savetxt('feature weight.txt', feature_names[coef_idx], delimiter='\n', fmt='%s') 
    online_pred(clf, X_new)
    print '-'*30
    
    clf_test(1, 2)
    clf_test(1, 3)
    clf_test(2, 1)
    clf_test(2, 3)
    clf_test(3, 1)
    clf_test(3, 2)
    clf_test(4, 3)
    clf_test(5, 2)
    clf_test(6, 1)
    