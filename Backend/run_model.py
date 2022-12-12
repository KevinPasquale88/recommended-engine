import keras
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop(["name"], axis=1)
    df = df.dropna()
    return df


SEED = 111
vocab_size = 40000
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

df = load_dataset("amazon_baby.csv")
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=SEED)
train_samples, train_labels = training_data["review"], training_data["rating"]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_samples)

@app.route("/model", methods = ['POST', 'GET'])
def main():
    data = request.get_json(force=True)
    sentence = data["sentence"]
    model_path = data["model_path"]
    dataset_path = data["dataset_path"]
    print(sentence)


    sentence = [sentence]
    sentence_sequence = tokenizer.texts_to_sequences(sentence)
    sentence_padded = pad_sequences(sentence_sequence, maxlen=max_length, truncating=trunc_type)

    model = keras.models.load_model(os.path.join(os.path.dirname(__file__), model_path))
    output = model.predict(sentence_padded)
    print(output)
    return jsonify(str(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)