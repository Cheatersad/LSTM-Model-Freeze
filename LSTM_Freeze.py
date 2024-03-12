#!/usr/bin/env python
# coding: utf-8

# In[1]:


#for data analysis and modeling
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing import text, sequence 
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#for text cleaning
import string
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#for visualization
import matplotlib.pyplot as plt


# In[2]:


true = pd.read_csv('True.csv')
true.head()


# In[3]:


fake = pd.read_csv('Fake.csv')
fake.head()


# In[4]:


true['truth'] = 1
fake['truth'] = 0
df = pd.concat([true, fake], axis=0, ignore_index=True)
df.shape


# In[5]:


get_ipython().run_cell_magic('time', '', 'def clean_text(txt):\n    """""\n    cleans the input text in the following steps\n    1- replace contractions\n    2- removing punctuation\n    3- spliting into words\n    4- removing stopwords\n    5- removing leftover punctuations\n    """""\n    contraction_dict = {"ain\'t": "is not", "aren\'t": "are not","can\'t": "cannot", "\'cause": "because", "could\'ve": "could have", \n                        "couldn\'t": "could not", "didn\'t": "did not",  "doesn\'t": "does not", "don\'t": "do not", "hadn\'t": "had not", \n                        "hasn\'t": "has not", "haven\'t": "have not", "he\'d": "he would","he\'ll": "he will", "he\'s": "he is",\n                        "how\'d": "how did", "how\'d\'y": "how do you", "how\'ll": "how will", "how\'s": "how is",  "I\'d": "I would", \n                        "I\'d\'ve": "I would have", "I\'ll": "I will", "I\'ll\'ve": "I will have","I\'m": "I am", "I\'ve": "I have", \n                        "i\'d": "i would", "i\'d\'ve": "i would have", "i\'ll": "i will",  "i\'ll\'ve": "i will have","i\'m": "i am", \n                        "i\'ve": "i have", "isn\'t": "is not", "it\'d": "it would", "it\'d\'ve": "it would have", "it\'ll": "it will", \n                        "it\'ll\'ve": "it will have","it\'s": "it is", "let\'s": "let us", "ma\'am": "madam", "mayn\'t": "may not",\n                        "might\'ve": "might have","mightn\'t": "might not","mightn\'t\'ve": "might not have", "must\'ve": "must have",\n                        "mustn\'t": "must not", "mustn\'t\'ve": "must not have", "needn\'t": "need not", "needn\'t\'ve": "need not have",\n                        "o\'clock": "of the clock", "oughtn\'t": "ought not", "oughtn\'t\'ve": "ought not have", "shan\'t": "shall not",\n                        "sha\'n\'t": "shall not", "shan\'t\'ve": "shall not have", "she\'d": "she would", "she\'d\'ve": "she would have", \n                        "she\'ll": "she will", "she\'ll\'ve": "she will have", "she\'s": "she is", "should\'ve": "should have", \n                        "shouldn\'t": "should not", "shouldn\'t\'ve": "should not have", "so\'ve": "so have","so\'s": "so as",\n                        "this\'s": "this is","that\'d": "that would", "that\'d\'ve": "that would have", "that\'s": "that is",\n                        "there\'d": "there would", "there\'d\'ve": "there would have", "there\'s": "there is", "here\'s": "here is",\n                        "they\'d": "they would", "they\'d\'ve": "they would have", "they\'ll": "they will", "they\'ll\'ve": "they will have",\n                        "they\'re": "they are", "they\'ve": "they have", "to\'ve": "to have", "wasn\'t": "was not", "we\'d": "we would", \n                        "we\'d\'ve": "we would have", "we\'ll": "we will", "we\'ll\'ve": "we will have", "we\'re": "we are",\n                        "we\'ve": "we have", "weren\'t": "were not", "what\'ll": "what will", "what\'ll\'ve": "what will have", \n                        "what\'re": "what are",  "what\'s": "what is", "what\'ve": "what have", "when\'s": "when is", \n                        "when\'ve": "when have", "where\'d": "where did", "where\'s": "where is", "where\'ve": "where have",\n                        "who\'ll": "who will", "who\'ll\'ve": "who will have", "who\'s": "who is", "who\'ve": "who have", \n                        "why\'s": "why is", "why\'ve": "why have", "will\'ve": "will have", "won\'t": "will not",\n                        "won\'t\'ve": "will not have", "would\'ve": "would have", "wouldn\'t": "would not",\n                        "wouldn\'t\'ve": "would not have", "y\'all": "you all", "y\'all\'d": "you all would",\n                        "y\'all\'d\'ve": "you all would have","y\'all\'re": "you all are","y\'all\'ve": "you all have",\n                        "you\'d": "you would", "you\'d\'ve": "you would have", "you\'ll": "you will", "you\'ll\'ve": "you will have",\n                        "you\'re": "you are", "you\'ve": "you have"}\n    def _get_contractions(contraction_dict):\n        contraction_re = re.compile(\'(%s)\' % \'|\'.join(contraction_dict.keys()))\n        return contraction_dict, contraction_re\n\n    def replace_contractions(text):\n        contractions, contractions_re = _get_contractions(contraction_dict)\n        def replace(match):\n            return contractions[match.group(0)]\n        return contractions_re.sub(replace, text)\n\n    # replace contractions\n    txt = replace_contractions(txt)\n    \n    #remove punctuations\n    txt  = "".join([char for char in txt if char not in string.punctuation])\n    txt = re.sub(\'[0-9]+\', \'\', txt)\n    \n    # split into words\n    words = word_tokenize(txt)\n    \n    # remove stopwords\n    stop_words = set(stopwords.words(\'english\'))\n    words = [w for w in words if not w in stop_words]\n    \n    # removing leftover punctuations\n    words = [word for word in words if word.isalpha()]\n    \n    cleaned_text = \' \'.join(words)\n    return cleaned_text\n    \ndf[\'data_cleaned\'] = df[\'title\'].apply(lambda txt: clean_text(txt))\n')


# In[6]:


df['data_cleaned']


# In[7]:


xtrain, xtest, ytrain, ytest = train_test_split(df['data_cleaned'], df['truth'], shuffle=True, test_size=0.2)
# find the length of the largest sentence in training data
max_len = xtrain.apply(lambda x: len(x)).max()
print(f'Max number of words in a text in training data: {max_len}')


# In[8]:


max_words = 10000
tokenizer = text.Tokenizer(num_words = max_words)
# create the vocabulary by fitting on x_train text
tokenizer.fit_on_texts(xtrain)
# generate the sequence of tokens
xtrain_seq = tokenizer.texts_to_sequences(xtrain)
xtest_seq = tokenizer.texts_to_sequences(xtest)

# pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)
word_index = tokenizer.word_index

print('text example:', xtrain[0])
print('sequence of indices(before padding):', xtrain_seq[0])
print('sequence of indices(after padding):', xtrain_pad[0])


# In[9]:


get_ipython().run_cell_magic('time', '', 'embedding_vectors = {}\n# with open(\'/kaggle/input/glove6b100d/glove.6B.100d.txt\',\'r\',encoding=\'utf-8\') as file:\nwith open(\'glove.42B.300d.txt\',\'r\',encoding=\'utf-8\') as file:\n    for row in file:\n        values = row.split(\' \')\n        word = values[0]\n        weights = np.asarray([float(val) for val in values[1:]])\n        embedding_vectors[word] = weights\nprint(f"Size of vocabulary in GloVe: {len(embedding_vectors)}")   \n')


# In[10]:


#initialize the embedding_matrix with zeros
emb_dim = 300
if max_words is not None: 
    vocab_len = max_words 
else:
    vocab_len = len(word_index)+1
embedding_matrix = np.zeros((vocab_len, emb_dim))
oov_count = 0
oov_words = []
for word, idx in word_index.items():
    if idx < vocab_len:
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            oov_count += 1 
            oov_words.append(word)
#print some of the out of vocabulary words
print(f'Some out of valubulary words: {oov_words[0:5]}')


# In[11]:


print(f'{oov_count} out of {vocab_len} words were OOV.')


# In[12]:


lstm_model = Sequential()
lstm_model.add(Embedding(vocab_len, emb_dim, trainable = False, weights=[embedding_matrix]))
lstm_model.add(LSTM(128, return_sequences=False))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(1, activation = 'sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(lstm_model.summary())


# In[13]:


get_ipython().run_cell_magic('time', '', 'batch_size = 256\nepochs  = 10\nhistory = lstm_model.fit(xtrain_pad, np.asarray(ytrain), validation_data=(xtest_pad, np.asarray(ytest)), batch_size = batch_size, epochs = epochs)\n')


# In[14]:


#plot accuracy
plt.figure(figsize=(15, 7))
plt.plot(range(epochs), history.history['accuracy'])
plt.plot(range(epochs), history.history['val_accuracy'])
plt.legend(['training_acc', 'validation_acc'])
plt.title('Accuracy')


# In[15]:


train_lstm_results = lstm_model.evaluate(xtrain_pad, np.asarray(ytrain), verbose=0, batch_size=256)
test_lstm_results = lstm_model.evaluate(xtest_pad, np.asarray(ytest), verbose=0, batch_size=256)
print(f'Train accuracy: {train_lstm_results[1]*100:0.2f}')
print(f'Test accuracy: {test_lstm_results[1]*100:0.2f}')


# In[17]:


tf.saved_model.save(lstm_model, 'saved_model/LSTM_model')


# In[18]:


lstm_model.save('lstm_model.h5')


# In[19]:


weights = lstm_model.get_weights()  
with open('lstm_model_weights.bin', 'wb') as f:
    for weight in weights:
        f.write(weight.tobytes()) 


# In[22]:


lstm_model.save('lstm_model_keras') 


# In[24]:


from tensorflow.keras.models import load_model

loaded_model_h5 = load_model('lstm_model.h5')

test_loss_h5, test_accuracy_h5 = loaded_model_h5.evaluate(xtest_pad, ytest, verbose=2)

print(f'Test accuracy using .h5 file: {test_accuracy_h5*100:.2f}%')


# In[28]:


import tensorflow as tf

loaded_model = tf.saved_model.load('saved_model/LSTM_model')

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def predict(input_data):
    return loaded_model.signatures['serving_default'](embedding_input=input_data)

xtest_pad_float = tf.cast(xtest_pad, tf.float32)

predictions = predict(xtest_pad_float)

test_accuracy = tf.keras.metrics.BinaryAccuracy()
test_accuracy.update_state(ytest, predictions['dense']) 
accuracy = test_accuracy.result().numpy()

print(f'Test accuracy using saved_model: {accuracy*100:.2f}%')


# In[30]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

def load_weights_from_bin(file_path, layer_shapes):
    weights = []
    with open(file_path, 'rb') as f:
        for shape in layer_shapes:
            num_elements = np.prod(shape)
            weight_flat = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
            weight_reshaped = weight_flat.reshape(shape)
            weights.append(weight_reshaped)
    return weights

vocab_len = 10000
emb_dim = 300

layer_shapes = [
    (vocab_len, emb_dim),
    (emb_dim, 512),
    (128, 512),
    (512,),
    (128, 1),
    (1,)
]

loaded_weights = load_weights_from_bin('lstm_model_weights.bin', layer_shapes)

lstm_model_from_weights = Sequential([
    Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model_from_weights.set_weights(loaded_weights)

lstm_model_from_weights.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:




