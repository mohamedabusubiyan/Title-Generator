import pandas as pd
import string
import numpy as np
import json

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
import keras.utils as ku
import tensorflow as tf
tf.random.set_seed(2)
from numpy.random import seed
seed(1)

df1 = pd.read_csv('USvideos.csv')
df2 = pd.read_csv('CAvideos.csv')
df3 = pd.read_csv('GBvideos.csv')

data1 = json.load(open('US_category_id.json'))
data2 = json.load(open('CA_category_id.json'))
data3 = json.load(open('GB_category_id.json'))

def category_extractor(data):
    i_d = [data['items'][i]['id'] for i in range(len(data['items']))]
    title = [data['items'][i]['snippet']["title"] for i in range(len(data['items']))]
    i_d = list(map(int, i_d))
    category = zip(i_d, title)
    category = dict(category)
    return category

df1['category_title'] = df1['category_id'].map(category_extractor(data1))
df2['category_title'] = df2['category_id'].map(category_extractor(data2))
df3['category_title'] = df3['category_id'].map(category_extractor(data3))

df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.drop_duplicates('video_id')

entertainment = df[df['category_title'] == 'Entertainment']['title']
entertainment = entertainment.tolist()

def clean_text(text):
    text = ''.join(e for e in text if e not in string.punctuation).lower()
    text = text.encode('utf8').decode('ascii', 'ignore')
    return text

corpus = [clean_text(e) for e in entertainment]

word_to_index = {}
index_to_word = {}
corpus_words = ' '.join(corpus).split()
unique_words = sorted(set(corpus_words))

for i, word in enumerate(unique_words):
    word_to_index[word] = i
    index_to_word[i] = word

sequences = []
seq_length = 10

for line in corpus:
    sequence = []
    for word in line.split():
        index = word_to_index.get(word)
        if index is not None:
            sequence.append(index)
    if len(sequence) > 1:
        for i in range(1, len(sequence)):
            sequences.append(sequence[max(0, i-seq_length):i+1])


max_sequence_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

predictors, label = padded_sequences[:,:-1], padded_sequences[:, -1]
label = ku.to_categorical(label, num_classes=len(word_to_index))

def create_model(max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


model = create_model(max_sequence_len, len(word_to_index))
model.fit(predictors, label, epochs=20, verbose=5)

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len, word_to_index, index_to_word):
    for _ in range(next_words):
        token_list = [word_to_index[word] for word in seed_text.split() if word in word_to_index]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        
        # Select the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        output_word = index_to_word.get(predicted_index, "")
        seed_text += " " + output_word
    return seed_text.title()


seed_text = input("Enter the seed text: ")

generated_text = generate_text(seed_text, 10, model, max_sequence_len, word_to_index, index_to_word)
print("Generated text:", generated_text)
