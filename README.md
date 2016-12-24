

## Twitter Sentiment Analysis using word embeddings and neural networks

This repository contains code to train a word embeddings model and then use neural network to learn and predict the sentiment label.
Code uses keras library with theano backend.
Having CUDA installed is recommended.

Dataset used is the one provided by Sentiment140.
Dataset link: http://help.sentiment140.com/for-students/

The dataset folder contains 4 text files:
1.Positive train
2.Positive test
3.Negative train
4.Negative test

Files contain raw text of tweet with the condition of one tweet per line. 
To use with own custom dataset, make sure you follow this structure, else make appropriate changes in code while loading data as per the label.
Keep the files in same folder as the code.

### Preprocessig:

Data pre-processing includes:
1.Tokenization
2.Stop word removal
3.Lemmatization
4.Removing punctuation
5.Ignore username and hyperlinks

### Approaches:

Two approaches are evaluated:

#### 1. Word2Vec with Long Short Term Memory (LSTM)
	
	To execute this code using GPU:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_word2vec.py

	Accuracy:76.28 on 80000 train and 20000 test

#### 2. Doc2Vec with Multi Layer Perceptron

	To execute this code using GPU:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp_doc2vec.py

	Accuracy:79.28 on 1.6 million train and 700 test
	Accuracy:69.37 on 80000 train and 20000 test



