import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
import nltk
model = Word2Vec.load('word2vec_model.bin')

data = pd.read_csv('transcripts.csv')

text_column_name = 'transcript'
data['processed_text'] = data[text_column_name].apply(lambda x: word_tokenize(x.lower()))

def calculate_average_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

data['transcript_vector'] = data['processed_text'].apply(calculate_average_vector)

transcript_vectors = np.vstack(data['transcript_vector'].to_numpy())

scaler = StandardScaler()
transcript_vectors_normalized = scaler.fit_transform(transcript_vectors)

num_clusters = 20

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(transcript_vectors_normalized)

data['cluster'] = clusters

data_sorted = data.sort_values(by='cluster')


output_file_path = 'transcript_clusters_kmeans_sorted.txt'
data_sorted[['transcript', 'cluster']].to_csv(output_file_path, index=False)

print(f"Transcript clusters (sorted) saved to: {output_file_path}")
