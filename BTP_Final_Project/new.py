import numpy as np
import pandas as pd
import pickle

def abusive_hinglish_to_english(data):
    hin=pd.read_csv("Hinglish_Profanity_List.csv")
    hin_bad_words = hin.iloc[:,0].values.tolist()
    bad_words_to_english = hin.iloc[:,1].values.tolist()
    hin = hin.iloc[:,:-1].values.tolist()
    cnt=0
    for sentence in data:
        wordList = sentence.split()
        for word in hin_bad_words:
            if word in wordList:
                x=wordList.index(word)
                wordList[x]=bad_words_to_english[hin_bad_words.index(word)]
        sentence = ' '.join(wordList)
        data[cnt]=sentence
        cnt+=1
    return data

def myinput_network(text):
    columns = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
    l=[text]
    l = abusive_hinglish_to_english(l);
    f='vect'
    vect= pickle.load(open(f, 'rb'))
    user_data = vect.transform(l)
    results2 = pd.DataFrame(columns=columns)
    mymodels={}
    for i in range(6):
        filename='model_'+str(i)
        mymodels[columns[i]]= pickle.load(open(filename, 'rb'))
    for i in range(6):
        user_results = mymodels[columns[i]].predict_proba(user_data)[:,1]
        results2[columns[i]] = user_results
    x = columns
    return results2.iloc[0].values,x

    