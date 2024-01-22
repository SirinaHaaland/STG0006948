import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string
import nltk


nltk.download('punkt')

def load_csv(transcripts):
    return pd.read_csv(transcripts)

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

def save_word2vec_model(model, model_path):
    model.save(model_path)

if __name__ == "__main__":
    csv_file_path = 'transcripts.csv'

    data = load_csv(csv_file_path)

    text_column_name = 'transcript'
    data['processed_text'] = data[text_column_name].apply(preprocess_text)

    sentences = data['processed_text'].tolist()
    model = train_word2vec_model(sentences)

    model_path = 'word2vec_model.bin'
    save_word2vec_model(model, model_path)
