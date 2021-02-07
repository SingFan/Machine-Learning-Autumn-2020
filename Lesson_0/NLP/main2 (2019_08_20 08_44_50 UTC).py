# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:30:31 2019

@author: Brian Chan
"""

# =============================================================================
# import and organize data
# =============================================================================

import pandas as pd
import numpy as np
Data_raw = pd.read_csv('reviews.csv')

#Data = Data_raw.loc[Data_raw['listing_id'] == '8637229',listing_id]

Data1 = pd.DataFrame([], columns = Data_raw.columns)
Data2 = pd.DataFrame([], columns = Data_raw.columns)
for i in np.arange(0,len(Data_raw)):
    #print(i)
    if (Data_raw.iloc[i,0] == 8637229):
        Data1 = Data1.append(Data_raw.iloc[i,:])
    elif (Data_raw.iloc[i,0] == 850531):
        Data2 = Data2.append(Data_raw.iloc[i,:])
    else:
        pass

print(len(Data1))
print(len(Data2))

import itertools
import scipy
  
import networkx as nx
import numpy as np
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer

def sentence_similarity(vector1, vector2):
    similarity_score = 1 - cosine_distance(vector1, vector2)

    if np.isnan(similarity_score):
        similarity_score = 0

    return similarity_score

def build_similarity_matrix(sentence_vectors, verbose=False):
    sentence_length = sentence_vectors.shape[0]

    if isinstance(sentence_vectors, scipy.sparse.csr.csr_matrix):
        sentence_vector_arrays = sentence_vectors.toarray()
    else:
        sentence_vector_arrays = sentence_vectors

    # Create an empty similarity matrix
    similarity_matrix = np.zeros((sentence_length, sentence_length))

    # create index of word pairs
    permutation_set = list(itertools.permutations(range(0, sentence_length), 2))

    for pair in permutation_set:
        idx1, idx2 = pair

        sent1 = sentence_vector_arrays[idx1]
        sent2 = sentence_vector_arrays[idx2]

        if verbose:
            print(f"Sentences: \n{sent1}\n{sent2}")

        similarity_matrix[idx1][idx2] = sentence_similarity(sentence_vector_arrays[idx1], sentence_vector_arrays[idx2])

    return similarity_matrix
  
def get_sentence_vector(word_embeddings, sentence, we_dim):
    vector = np.zeros((we_dim, ))

    sentence_length = len(sentence) + 0.001

    if sentence:
        sentence_embeddings = sum([word_embeddings.get(word, vector) for word in sentence])
        vector = sentence_embeddings/sentence_length

    return vector

def visualise_top_scores(scores, flatten_sentences, n_top):
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(flatten_sentences)), reverse=True)    

    summarize_text = []

    for i in range(n_top):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return summarize_text

sentence_embedding_vectors = []

for review_sentence in flatten_review_sentences:
    sentence_vector = get_sentence_vector(word_embeddings, review_sentence, dims)
    sentence_embedding_vectors.append(sentence_vector)

glove_similarity_matrix = build_similarity_matrix(np.array(sentence_embedding_vectors))
glove_sentence_similarity_graph = nx.from_numpy_array(glove_similarity_matrix)
glove_scores = nx.pagerank(glove_sentence_similarity_graph)

top_n = 5
visualise_top_scores(scores=glove_scores, flatten_sentences=flatten_review_sentences, n_top=top_n)