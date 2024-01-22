import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

model = Word2Vec.load('word2vec_model.bin')

word_vectors = [model.wv[word] for word in model.wv.index_to_key]

word_vectors_array = np.array(word_vectors)

scaler = StandardScaler()
word_vectors_array_normalized = scaler.fit_transform(word_vectors_array)

dbscan = DBSCAN(eps=10, min_samples=1000) 
clusters = dbscan.fit_predict(word_vectors_array_normalized)

word_clusters = {word: cluster for word, cluster in zip(model.wv.index_to_key, clusters)}

output_file_path = 'word_clusters.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for word, cluster in word_clusters.items():
        output_file.write(f"Word: {word}, Cluster: {cluster}\n")

print(f"Word clusters saved to: {output_file_path}")

