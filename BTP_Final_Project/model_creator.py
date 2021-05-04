#IMPORTING LIBRARIES AND FILES
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from os.path import join, dirname, realpath
import io
import re
import nltk
from nltk.stem import WordNetLemmatizer  
if __name__=='__main__':
    #TEXT PREPROCESSING
    data = pd.read_csv("train.csv")
    columns = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
    for i in range(1, 159570):
        line=re.sub('[^a-zA-Z]',' ',data['text'][i])
        line=line.lower()
        line=line.split()
        lm = WordNetLemmatizer()
        line=[lm.lemmatize(word) for word in line]
        line=' '.join(line)
        data['text'][i]=line
    
    train, test, = train_test_split(train, test_size=0.2)
    labels = train.iloc[:,2:]
    train_data = train.iloc[:,1]
    test_data = test.iloc[:,1]
    features = 5000
    ngram = (1,2)
   #TFIDF VECTORIZER(bag of words)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram, max_features=features)
    train_features = vectorizer.fit_transform(train_data)
    filename='vect'
    pickle.dump(vectorizer, open(filename, 'wb'))
    test_features = vectorizer.transform(test_data)
    logreg = LogisticRegression(C=10,solver="liblinear")
    models={}
    logistic_results = pd.DataFrame(columns=columns)    
    cnt=0

    from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
    avg_acc = 0
    avg_prec=0
    avg_rec=0
    model_acc = []
    model_prec= []
    model_rec= []

    for i in columns:
        y = train[i]
        models[i]=copy.copy(logreg.fit(train_features, y))
        filename = "model_"+ str(cnt)
        pickle.dump(models[i], open(filename, 'wb'))
        ypred_X = logreg.predict(train_features)
        acc = accuracy_score(log_test[i], ypred_X)
        prec=precision_score(log_test[i], ypred_X)
        rec=recall_score(log_test[i], ypred_X)
        avg_acc = avg_acc + acc
        avg_prec = avg_prec + prec
        avg_rec = avg_rec + rec
        print("\n Accuracy score for category "+str(i)+" using Logistic Regression: "+str(100*acc)+"%")
        print("\n Precision score for category "+str(i)+" using Logistic Regression: "+str(100*prec)+"%")
        print("\n Recall score for category "+str(i)+" using Logistic Regression: "+str(100*rec)+"%")
        testy_prob = logreg.predict_proba(test_features)[:,1]
        logistic_results[i] = testy_prob
        cnt+=1

        print("\n Average Accuracy Score for Logistic Regression: "+str(100*avg_acc/6)+"\n")
        print("\n Average Precision Score for Logistic Regression: "+str(100*avg_prec/6)+"\n")
        print("\n Average Recall Score for Logistic Regression: "+str(100*avg_rec/6)+"\n")

def myinput_network(text):
    columns = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
    l=[text]
    f='vect'
    vect= pickle.load(open(f, 'rb'))
    user_data = vect.transform(l)
    results2 = pd.DataFrame(columns=columns)
    cnt=0
    mymodels={}
    for i in range(6):
        filename='model_'+str(i)
        mymodels[columns[i]]= pickle.load(open(filename, 'rb'))
    for i in range(6):
        user_results = mymodels[columns[i]].predict_proba(user_data)[:,1]
        results2[columns[i]] = user_results
    x = columns
    return results2.iloc[0].values,x
