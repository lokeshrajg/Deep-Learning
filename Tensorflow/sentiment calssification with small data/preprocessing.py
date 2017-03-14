### Preprocessing the text into vector of features to build the classifier for sentiment analysis for data of around 10k text sentences
# Libraries
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

### the text is converted into bag of words using nltk
lemmitizer=WordNetLemmatizer()
hm_lines=100000

#### creates the list of unique words from the text in pos.txt and neg.txt
def create_lexicon(pos,neg):
    lexicon=[]
    with open(pos,'r') as f:
        content = f.readlines()
        for l in content[:hm_lines]:
            words= word_tokenize(l)
            lexicon +=list(words)

    with open(neg,'r') as f:
        content = f.readlines()
        for l in content[:hm_lines]:
            words= word_tokenize(l)
            lexicon +=list(words)

    lexicon= [lemmitizer.lemmatize(i) for i in lexicon]
    w_count= Counter(lexicon)
    l_word=[]
    for l in w_count:
        if 1000>w_count[l]>50:
            l_word.append(l)
    print(len(l_word))
    return l_word


### creates feature which is a list of length equal to the number of unique words in lexicon and 
# update the index of the elements of the list if the word in the text is present in the lexicon
def bag_of_words(sample,lexicon,classification):
    feature=[]
    with open(sample,'r') as f:
        content = f.readlines()
        for l in content[:hm_lines]:
            words= word_tokenize(l.lower())
            words = [lemmitizer.lemmatize(i) for i in words]
            featureset= np.zeros(len(lexicon))
            for i in words:
                if i.lower() in lexicon:
                    index= lexicon.index(i.lower())
                    featureset[index]+=1

            featureset=list(featureset)
            feature.append([featureset,classification])

    return feature

# training and testing features and labels
def create_pos_and_neg(pos,neg,test_size=0.1):
    lexicon=create_lexicon(pos,neg)
    feature=[]
    feature+= bag_of_words('pos.txt',lexicon,[1,0])
    feature += bag_of_words('neg.txt', lexicon, [0, 1])
    random.shuffle(feature)
    feature =np.array(feature)

    testing_size= int(test_size*len(feature))

    train_x = list(feature[:, 0][:-testing_size])
    train_y = list(feature[:, 1][:-testing_size])
    test_x = list(feature[:, 0][-testing_size:])
    test_y = list(feature[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__=='__main__':
    train_x,train_y,test_x,test_y = create_pos_and_neg("pos.txt","neg.txt")
    with open('sentiment.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
