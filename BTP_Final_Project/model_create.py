import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import copy
import os
import pickle

if __name__=='__main__':
        train = pd.read_csv("train.csv")
        columns = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
        train, test = train_test_split(train, test_size=0.2)
        labels = train.iloc[:,2:]
        train_data = train.iloc[:,1]
        test_data = test.iloc[:,1]

        features = 5000
        ngram = (1,2)
        vectorizer = TfidfVectorizer(stop_words='english',\
                                token_pattern = "\w*[a-z]\w*",\
                                ngram_range=ngram,\
                                max_features=features)

        train_features = vectorizer.fit_transform(train_data)
        filename='vect'
        pickle.dump(vectorizer, open(filename, 'wb'))
        test_features = vectorizer.transform(test_data)
        logreg = LogisticRegression(C=10,solver="liblinear")
        models={}
        logistic_results = pd.DataFrame(columns=columns)    
        cnt=0
        
        for i in columns:
                y = train[i]
                models[i]=copy.copy(logreg.fit(train_features, y))
                filename = "model_"+ str(cnt)
                pickle.dump(models[i], open(filename, 'wb'))
                ypred_X = logreg.predict(train_features)
                testy_prob = logreg.predict_proba(test_features)[:,1]
                logistic_results[i] = testy_prob
                cnt+=1