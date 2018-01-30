from gensim.models.keyedvectors import KeyedVectors
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from random import randint

def getvalueofnode(node):
    return node.text if node is not None else None

def readStopWords():
    words = []
    myFile = open('stopword.txt', 'r')
    for line in myFile:
        word = line.split()[0]
        words.append(word)
    return words

stopwords = readStopWords();


def tokenizer(text):
    #print(text)
    text = str(text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text).replace('-', '')
    text = re.sub('<[^>]*>', '', text).replace('_', '')
    text = re.sub('[^|][0-9]+', ' ', text)
    text = re.sub('[\s!/,\\.?¡¿"“”:/();]+', ' ', text)
    text = re.sub(r'(?:@[\w_]+)', '', text)
    text = re.sub('[W]+', ' ', text)
    text = re.sub('[\s]+', ' ', text.lower())
    text = re.sub(r'\b(gg+\b)+', 'por', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub('^\s', '', text)
    text = re.sub('@([a-z0-9_]+)', '@user', text)
    text = re.sub(r'\b(x\b)+', 'por', text)
    text = re.sub(r'\b(q\b)+|\b(k\b)+|\b(ke\b)+|\b(qe\b)+|\b(khe\b)+|\b(kha\b)+', 'que', text)
    text = re.sub(r'\b(xk\b)+|\b(xq\b)+', 'porque', text)
    text = re.sub(r'\b(xd\b)+|\b(xq\b)+', 'porque', text)
    text = re.sub(r'\b(ai+uda\b)+', 'ayuda', text)
    text = re.sub(r'\b(hostia\b)+', 'ostia', text)
    text = re.sub(r'\b(d\b)+', 'de', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text).replace('-', '')
    text = re.sub(r'\b(d*xd+x*[xd]*\b|\ba*ha+h[ha]*\b|\ba*ja+j*[ja]*|o?l+o+l+[ol])\b', 'ja', text)
    text = [w for w in text.split() if w not in stopwords]
    #print(text)
    return text

def loadData(fileNames):
    #parsedXML = et.parse( "intertass-train-tagged.xml" )
    #parsedXML = et.parse( "intertass-development-tagged.xml")
    texts = []
    dfcols = ['content', 'aspect','polarity']
    df_xml = pd.DataFrame(columns=dfcols)
    for fileName in fileNames:
        parsedXML = et.parse( fileName)
        for node in parsedXML.getroot():
            #node = parsedXML.getroot()[0]
            content = node.findall('sentiment')
            aspects = []
            polarity = []
            for value in content:
                value.text = value.get('aspect')
                aspects.append(value.get('aspect'))
                polarity.append(value.get('polarity'))
                #print((node.itertext()))
            text = ''
            for value in node.itertext():
                text = text + value
            text = tokenizer(text)
            for i in range(len(aspects)):
                aspects[i] = tokenizer(aspects[i])[0]

            p = []
            k = 0
            #print(aspects)
            for i in range(len(text)):
                word = text[i]
                if word == aspects[k]:
                    p.append(i)
                    k = k + 1
                    if k == len(aspects):
                        break
            #print(text)
            texts.append(text)
            #print(p)
            for i in range(len(p)):
                a = max(p[i]-2, 0)
                b = min(p[i]+4, len(text))
                textEnd = text[a:b]
                df_xml = df_xml.append(
                pd.Series([textEnd, aspects[i], polarity[i]], index=dfcols), ignore_index=True)
            #df_xml = df_xml.append(pd.Series([text, aspects, polarity], index=dfcols), ignore_index=True)
                #print((value)
    return df_xml

data_imput = loadData(["socialtv-train-tagged.xml"])
print(data_imput)

X_train = list()
Y_train = list()
Z_train = list()
ranNom = np.arange(data_imput.shape[0])
np.random.shuffle(ranNom)
for i in range(data_imput.shape[0]):
    X_train.append((data_imput.at[(ranNom[i], 'content')]))
    Y_train.append(data_imput.at[(ranNom[i], 'polarity')])
    Z_train.append(data_imput.at[(ranNom[i], 'aspect')])

    #print(data_imput.at[(ranNom[i], 'polarity')])


modWord2vec = Word2Vec(X_train, size=50, window=5, min_count=5, workers=4)
word_vectors = modWord2vec.wv
word_vectors.save_word2vec_format("Word2Vec2.txt")
