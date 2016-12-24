
from __future__ import print_function
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import 	re
from gensim.corpora.dictionary import Dictionary
import multiprocessing

from random import shuffle

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.recurrent import LSTM

np.random.seed(1500)  # For Reproducibility

import logging
import sys


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


maxlen = 300  # cut texts after this number of words 
batch_size = 64

# If you want to load a pre trained doc2vec model
#print('Loading data...')

#model = Doc2Vec.load('./18_6_2.d2v')
#print(len(model.vocab))

def clean_tweet(tweet):
	lemmatizer=WordNetLemmatizer()
	word_list = tweet.split()
	mod_tweet = []
	for i in word_list:
		i = unicode(i,errors='ignore')
		if '@'  in i or 'http:' in i:
			continue
		j = re.sub(r'(.)\1+',r'\1\1',i)
		mod_tweet.append(lemmatizer.lemmatize(j))
	return mod_tweet


log = logging.getLogger()
#log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(LabeledSentenceclean_tweet(line), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(clean_tweet(line), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences
        

log.info('source load')
sources = {'negative_test.txt':'TEST_NEG', 'positive_test.txt':'TEST_POS', 'negative_train.txt':'TRAIN_NEG', 'positive_train.txt':'TRAIN_POS'}


log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(window=5, size=300, sample=1e-3, workers=8,hs=1,dbow_words=1)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	print ('Epoch: ' + str(epoch))
	model.train(sentences.sentences_perm())


#TO SAVE DOC2VEC Model
#log.info('Model Save')
#model.save('./11_5.d2v')

#model = Doc2Vec.load('./11_5.d2v')

#log.info('Sentiment')
train_arrays = np.zeros((80000, 300))
train_labels = np.zeros(80000)

for i in range(40000):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[40000 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[40000 + i] = 0


test_arrays = np.zeros((20000, 300))
test_labels = np.zeros(20000)

for i in range(10000):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[10000 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[10000 + i] = 0


print('Building model...')


mlp_model = Sequential()
mlp_model.add(Dense(512, input_shape=(maxlen,)))
mlp_model.add(Activation('relu'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(1))
mlp_model.add(Activation('sigmoid'))	

# try using different optimizers and different optimizer configs
mlp_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training...')

mlp_model.fit(train_arrays, train_labels, batch_size=batch_size, nb_epoch=5,
          validation_data=(test_arrays, test_labels))
score, acc = mlp_model.evaluate(test_arrays, test_labels,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)


