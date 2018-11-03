from __future__ import print_function
import collections
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import tensorflow as tf
import pickle
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, Multiply, Merge, Dot
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import gensim
import json

model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

#model = dict()

with open('./Data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('./Data/labels.txt', 'r') as f:
    labels_org = f.read()

from string import punctuation
	
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()


#changing here
words = list(set(words))
vocab_to_int = dict()

for i in range(len(words)):
 vocab_to_int.update({words[i]:i})
#from collections import Counter
#counts = Counter(words)
#vocab = sorted(counts, key=counts.get, reverse=True)
#vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])

labels = np.array([1 if l == "positive" else 0 for l in labels_org.split()])

from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# Filter out that review with 0 length
reviews_ints = [r[0:200] for r in reviews_ints if len(r) > 0]

from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)
# print(features[:10,:100])
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
features[:10,:100]

split_frac = 0.8

split_index = int(split_frac * len(features))

train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape),
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))

lstm_size = 256
lstm_layers = 2
batch_size = 1
learning_rate = 0.003

n_words = len(vocab_to_int) + 1 # Add 1 for 0 added to vocab

# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

embed_size = 300

w2v_embed = np.ndarray([n_words,embed_size])


for i in range(n_words-1):
 if words[i] not in model:
   w2v_embed[vocab_to_int[words[i]]] = np.array([0]*embed_size)
 else:
   w2v_embed[vocab_to_int[words[i]]] = model[words[i]]



import random
'''
idx = random.sample(range(len(train_x)), 1000)

train_x_s = []
train_y_s = []

for i in idx:
    train_x_s.append(train_x[i])
    train_y_s.append(train_y[i])

train_x = np.array(train_x_s)
train_y = np.array(train_y_s)
test_x = np.array(test_x)
test_y = np.array(test_y)
'''

train_x_e = np.ndarray((len(train_x), seq_len, embed_size))

for i in range(len(train_x)):
    for j in range(seq_len):
        train_x_e[i][j][:] = w2v_embed[train_x[i][j]]

val_x_e = np.ndarray((len(val_x), seq_len, embed_size))

for i in range(len(val_x)):
    for j in range(seq_len):
        val_x_e[i][j][:] = w2v_embed[val_x[i][j]]





hidden_size = 256
use_dropout = True
#vocabulary = n_words


model1 = Sequential()
#model1.add(Embedding(vocabulary, embed_size, input_length=mx_sent))
model1.add(LSTM(embed_size, return_sequences=True, input_shape=(seq_len, embed_size)))
model1.add(LSTM(embed_size, return_sequences=False))
if use_dropout:
    model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid', name='out1'))


optimizer = Adam()
# model1.compile(loss='mean_squared_error', optimizer='adam')
# parallel_model = multi_gpu_model(model, gpus=2)
parallel_model = model1
parallel_model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['acc'])
#parallel_model.compile(loss='mean_squared_error', optimizer='adam')


print(model1.summary())
#plot_model(model1,to_file='demo.png',show_shapes=True)
num_epochs = 50

parallel_model.fit(x= train_x_e, y=train_y, batch_size=1000, epochs=num_epochs,
               validation_split=0.1)

print(parallel_model.evaluate([val_sent_e,val_claim_e],val_y))
parallel_model.save("final_model.hdf5")
