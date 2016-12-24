
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
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout

np.random.seed(1500)  # For Reproducibility

import logging
import sys


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import 	re


def tokenizer(text):
    text = [clean_tweet(document) for document in text]
    return text

def clean_tweet(tweet):
	lemmatizer=WordNetLemmatizer()
	word_list = tweet.split()
	filtered_words=[word for word in word_list if word not in stopwords.words('english')]
	#repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
	#repl = r'\1\2\3'
	mod_tweet = []
	for i in filtered_words:
		i = unicode(i,errors='ignore')
		i.lower()
		i.strip('#\'"?,.!')
		if '@'  in i or 'http:' in i:
			continue
		j = re.sub(r'(.)\1+',r'\1\1',i)
		mod_tweet.append(lemmatizer.lemmatize(j))
	return mod_tweet


def sentences_perm(sentences):
    shuffle(sentences)
    return sentences


log = logging.getLogger()
#log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

vocab_dim = 118717
maxlen = 50
n_iterations = 10  # ideally more..
n_exposures = 30
window_size = 5
batch_size = 32
n_epoch = 2
input_length = 50
cpu_count = multiprocessing.cpu_count()


log.info('source load')
sources = {'negative_test.txt':'TEST_NEG', 'positive_test.txt':'TEST_POS', 'negative_train.txt':'TRAIN_NEG', 'positive_train.txt':'TRAIN_POS'}

def import_tag(datasets=None):
    if datasets is not None:
        train = {}
        test = {}
        for k, v in datasets.items():
            with open(k) as fpath:
                data = fpath.readlines()
            for val, each_line in enumerate(data):
                if v.endswith("NEG") and v.startswith("TRAIN"):
                    train[val] = each_line
                elif v.endswith("POS") and v.startswith("TRAIN"):
                    train[val + 40000] = each_line
                elif v.endswith("NEG") and v.startswith("TEST"):
                    test[val] = each_line
                else:
                    test[val + 10000] = each_line
        return train, test
    else:
        print('Data not found...')

def create_dictionaries(train=None,test=None,model=None):
    if (train is not None) and (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data):
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data
        train = parse_dataset(train)
        test = parse_dataset(test)
        return w2indx, w2vec, train, test
    else:
        print('No data provided...')


print('Loading Data...')
train, test = import_tag(datasets=sources)
combined = train.values() + test.values()

print('Tokenising...')
combined = tokenizer(combined)

#print combined

print('Training a Word2vec model...')
model = Word2Vec(size=maxlen,
                 window=window_size,
                 workers=cpu_count)

model.build_vocab(combined)

for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences_perm(combined))

print('Transform the Data...')
index_dict, word_vectors, train, test = create_dictionaries(train=train,
                                                            test=test,
                                                            model=model)

print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols, maxlen))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]

print('Creating Datesets...')
X_train = train.values()
y_train = [1 if value > 40000 else 0 for value in train.keys()]
X_test = test.values()
y_test = [1 if value > 10000 else 0 for value in test.keys()]

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert labels to Numpy Sets...')
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Defining a Simple Keras Model...')
lstm_model = Sequential()  # or Graph 
lstm_model.add(Embedding(output_dim=maxlen,
                    input_dim=n_symbols,
                    mask_zero=True,
                    weights=[embedding_weights],
                    input_length=input_length))  # Adding Input Length
lstm_model.add(LSTM(maxlen))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(1, activation='sigmoid'))

print('Compiling the Model...')
lstm_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
	      class_mode='binary')

print("Train...")
lstm_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
          validation_data=(X_test, y_test))

print("Evaluate...")
score, acc = lstm_model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
