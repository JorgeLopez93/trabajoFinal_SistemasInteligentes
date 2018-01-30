import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
from random import randint

maxSeqLength = 0
numDimensions = 50

def getvalueofnode(node):
    return node.text if node is not None else None

def loadData(fileNames):
    dfcols = ['content', 'aspect','polarity']
    df_xml = pd.DataFrame(columns=dfcols)
    for fileName in fileNames:
        parsedXML = et.parse( fileName)
        for node in parsedXML.getroot():
            content = node.findall('sentiment')
            aspects = []
            polarity = []
            for value in content:
                value.text = value.get('aspect')
                aspects.append(value.get('aspect'))
                polarity.append(value.get('polarity'))
            text = ''
            for value in node.itertext():
                text = text + value
            text = tokenizer(text)
            for i in range(len(aspects)):
                aspects[i] = tokenizer(aspects[i])[0]
            p = []
            k = 0
            for i in range(len(text)):
                word = text[i]
                if word == aspects[k]:
                    p.append(i)
                    k = k + 1
                    if k == len(aspects):
                        break
            for i in range(len(p)):
                a = max(p[i]-2, 0)
                b = min(p[i]+4, len(text))
                textEnd = text[a:b]
                df_xml = df_xml.append(
                pd.Series([textEnd, aspects[i], polarity[i]], index=dfcols), ignore_index=True)
    return df_xml

def readStopWords():
    words = []
    myFile = open('stopword.txt', 'r')
    for line in myFile:
        word = line.split()[0]
        words.append(word)
    return words

stopwords = readStopWords();

def readFile():
    new_dict = {}
    words = []
    myFile = open('Word2Vec.txt', 'r')
    for line in myFile:
        word = line.split()[0]
        words.append(word)
        vec = [float(i) for i in line.split()[1:51]]
        new_dict[word] = vec
    return new_dict, words
dict, wordsDic = readFile()

def tokenizer(text):

    text = str(text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text).replace('-', '')
    text = re.sub('<[^>]*>', '', text).replace('_', '')
    text = re.sub('[^|][0-9]+', ' ', text)
    text = re.sub('[\s!/,\\.?¡¿"“”:/();]+', ' ', text)
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
    return text

data_imput = loadData(["socialtv-train-tagged.xml"])
print(data_imput)


X_train = list()
Y_train = list()
Z_train = list()
ranNom = np.arange(data_imput.shape[0])

for i in range(data_imput.shape[0]):
    X_train.append((data_imput.at[(ranNom[i], 'content')]))
    Y_train.append(data_imput.at[(ranNom[i], 'polarity')])
    Z_train.append(data_imput.at[(ranNom[i], 'aspect')])


for x in X_train:
    #print(x)
    if len(x) > maxSeqLength:
        maxSeqLength = len(x)
print(maxSeqLength)


def get_minibatch(size):
    docs, y = [], []
    level_size = [0,0,0]
    for i in range(size):

        x_t = X_train[i]
        vec = []
        for j in range(len(x_t)):
            if x_t[j] in dict:
                vec.append(dict[x_t[j]])
        k = len(vec)
        for value in range(k, maxSeqLength):
            vec.append([0 for i in range(numDimensions) ])
        docs.append(vec)
        if Y_train[i] == 'P':
            y.append([1,0,0])
            level_size[0] = level_size[0] + 1;
        elif Y_train[i] == 'NEU':
            y.append([0,1,0])
            level_size[1] = level_size[1] + 1;
        elif Y_train[i] == 'N':
            y.append([0,0,1])
            level_size[2] = level_size[2] + 1;
    print(level_size)
    return docs, y


learning_rate = 0.01

lstmUnits = 60
numClasses = 3

training_steps = 10000
display_step = 10

def BiRNN(x, weights, biases):

    x = tf.unstack(x, maxSeqLength, 1)
    '''
    fw_cell = rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0)
    fw_cell = rnn.DropoutWrapper(cell=fw_cell, output_keep_prob=1.0)
    # Backward direction cell
    bw_cell = rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0)
    bw_cell = rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=1.0)
    '''
    fw_cell = rnn.GRUCell(lstmUnits)
    fw_cell = rnn.DropoutWrapper(cell=fw_cell, output_keep_prob=0.5)

    bw_cell = rnn.GRUCell(lstmUnits)
    bw_cell = rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=0.5)

    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

T = 3000
X_, Y_ = get_minibatch(3441)
X_train = X_[0:T]
y_train = Y_[0:T]
a = int(T/100)
X_test = X_[T:3441]
y_test = Y_[T:3441]

accMax = 0.45

tf.reset_default_graph()
X = tf.placeholder("float", [None, maxSeqLength, numDimensions])
Y = tf.placeholder("float", [None, numClasses])
weights = {

'out': tf.Variable(tf.random_normal([2*lstmUnits, numClasses]))
}
biases = {
'out': tf.Variable(tf.random_normal([numClasses]))
}
logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(init)
    k = -1
    loss = np.zeros(a)
    acc = np.zeros(a)

    accTestAnt = 1;
    for step in range(1, training_steps+1):

        start = 0
        end = 100
        k = k + 1
        if(k >= a):
            k = 0

        for i in range(a):

            batch_x = X_train[start:end]
            batch_y = y_train[start:end]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            start = end
            end = start + 100
        batch_x = X_train[k*100:(k + 1)*100]
        batch_y = y_train[k*100:(k + 1)*100]
        loss, acc = sess.run([loss_op, accuracy], feed_dict = {X: batch_x, Y: batch_y})

        accTest = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})

        if step % display_step == 0 or step == 1:
            print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Training Accuracy= " + \
            "{:.3f}".format(acc) + \
            ", Testing Accuracy:", accTest)


        if (accTest > accMax):
            accMax = accTest * 1
            save_path = saver.save(sess, "models/pretrained_lstm" + str(accMax) +".ckpt", global_step = step)
            print("saved to %s" % save_path)

    sess.close()
    print("Optimization Finished!")
