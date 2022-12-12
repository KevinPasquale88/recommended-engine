import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


SEED = 111
vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'

def load_dataset():
    df = pd.read_csv("amazon_baby.csv")
    df = df.drop(["name"], axis=1)
    df = df.dropna()
    return df

df = load_dataset()
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=SEED)
train_samples, train_labels = training_data["review"], training_data["rating"]
test_samples, test_labels = testing_data["review"], testing_data["rating"]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_samples)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_samples)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

test_sentences = tokenizer.texts_to_sequences(test_samples)
test_padded = pad_sequences(test_sentences, maxlen=max_length)

onehotencoder = OneHotEncoder(categories='auto')

train_labels = np.array(train_labels)
training_labels = np.zeros((train_labels.shape[0], 5))
print(test_padded)
for i in range(train_labels.shape[0]):
    training_labels[i, train_labels[i]-1] = 1
test_labels = np.array(test_labels)
testing_labels = np.zeros((test_labels.shape[0], 5))
for i in range(test_labels.shape[0]):
    testing_labels[i, test_labels[i]-1] = 1


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 20
history = model.fit(train_padded, training_labels, epochs=num_epochs, validation_data=(test_padded, testing_labels))
model.save("./model_saved/model")
np.save("history.npy", history.history)