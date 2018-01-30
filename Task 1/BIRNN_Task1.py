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
<
    dfcols = ['content', 'polarity']
    df_xml = pd.DataFrame(columns=dfcols)
    for fileName in fileNames:
        parsedXML = et.parse( fileName)
        for node in parsedXML.getroot():
            content = node.find('content')
            polarity = node.find('sentiment/polarity/value')
            if getvalueofnode(polarity) is None:
                polarity = node.find('sentiments/polarity/value')
            df_xml = df_xml.append(
                pd.Series([getvalueofnode(content), getvalueofnode(polarity)], index=dfcols), ignore_index=True)
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
print(dict['no'])
print(dict['si'])

def tokenizer(text):
    text = str(text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('\d', '', text)
    text = re.sub(r'(?:@[\w_]+)', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = re.sub(r'\b-\b', ' ', text)
    text = re.sub('[^|][0-9]+', ' ', text)
    text = re.sub('[\s!/,\\.?¡¿"“”:/();]+', ' ', text) #”
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
    text = re.sub(r'\b(d\b)+', 'de', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text).replace('-', '')
    text = re.sub(r'\b(d*xd+x*[xd]*\b|\ba*ha+h[ha]*\b|\ba*ja+j*[ja]*|o?l+o+l+[ol])\b', 'ja', text)
    text = [w for w in text.split() if w not in stopwords]
    text = [w for w in text if w in wordsDic]
    return text

data_imput = loadData(["intertass-train-tagged.xml", "intertass-development-tagged.xml"])

X_train = list()
Y_train = list()
ranNom = np.arange(data_imput.shape[0])
np.random.shuffle(ranNom)
for i in range(data_imput.shape[0]):
    X_train.append(tokenizer(data_imput.at[(ranNom[i], 'content')]))
    Y_train.append(data_imput.at[(ranNom[i], 'polarity')])

for x in X_train:
    if len(x) > maxSeqLength:
        maxSeqLength = len(x)
print(maxSeqLength)


def get_minibatch(size):
    docs, y = [], []
    level_size = [0,0,0,0]
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
            y.append([1,0,0,0])
            level_size[0] = level_size[0] + 1;
        elif Y_train[i] == 'NEU':
            y.append([0,1,0,0])
            level_size[1] = level_size[1] + 1;
        elif Y_train[i] == 'N':
            y.append([0,0,1,0])
            level_size[2] = level_size[2] + 1;
        elif Y_train[i] == 'NONE':
            y.append([0,0,0,1])
            level_size[3] = level_size[3] + 1;
    print(level_size)
    return docs, y

T = 1300
X_, Y_ = get_minibatch(1514)
X_train = X_[0:T]
y_train = Y_[0:T]
a = int(T/100)
X_test = X_[T:1500]
y_test = Y_[T:1500]

learning_rate = 0.2

batchSize = 256
lstmUnits = 32
numClasses = 4

training_steps = 10000
display_step = 10

tf.reset_default_graph()

X = tf.placeholder("float", [None, maxSeqLength, numDimensions])
Y = tf.placeholder("float", [None, numClasses])

weights = {
    'out': tf.Variable(tf.random_normal([2*lstmUnits, numClasses]))
}
biases = {
    'out': tf.Variable(tf.random_normal([numClasses]))
}


def BiRNN(x, weights, biases):

    x = tf.unstack(x, maxSeqLength, 1)

    fw_cell = rnn.GRUCell(lstmUnits)
    fw_cell = rnn.DropoutWrapper(cell=fw_cell, output_keep_prob=0.5)

    bw_cell = rnn.GRUCell(lstmUnits)
    bw_cell = rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=0.5)

    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x,
                                              dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x,
                                        dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(init)
    test_len = 128
    k = -1
    loss = np.zeros(a)
    acc = np.zeros(a)
    accMax = 0.45
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
    writer.close()
    sess.close()
    print("Optimization Finished!")
