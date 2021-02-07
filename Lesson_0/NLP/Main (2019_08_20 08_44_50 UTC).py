# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:14:29 2019

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


# =============================================================================
# Method 1: LDA
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx+1}:")
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def learning_by_LDA(Data, number_topics, number_words):
    # Create TF-IDF vectors from unigrams
    count_vectorizer = TfidfVectorizer(
                            strip_accents='unicode',
                            preprocessor=None,
                            analyzer='word',
                            ngram_range=(1, 1),
                            min_df=10,
                            use_idf=True, smooth_idf=True, 
                            max_features = 5000)
    #count_vectorizer.fit(list(Data['comments']))
    count_vectorizer.fit(Data['comments'].values.astype('U'))
    # LDA
    bag_of_words = count_vectorizer.transform(Data['comments'].values.astype('U'))
    
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, random_state=19, n_jobs=-1, learning_method='online')
    lda.fit(bag_of_words)
    
    print_topics(lda, count_vectorizer, number_words)
    return lda, count_vectorizer, number_words

number_topics = 5
number_words  = 5

print('Data1 | ', 'number_topics:', number_topics, 'number_words:', number_words)

lda, count_vectorizer, number_words = learning_by_LDA(Data1,number_topics, number_words)
print('\n')
print('Data2 | ', 'number_topics:', number_topics, 'number_words:', number_words)

lda, count_vectorizer, number_words = learning_by_LDA(Data2,number_topics, number_words)
print('\n')


number_topics = 5
number_words  = 10

print('Data1 | ', 'number_topics:', number_topics, 'number_words:', number_words)

lda, count_vectorizer, number_words = learning_by_LDA(Data1,number_topics, number_words)
print('\n')
print('Data2 | ', 'number_topics:', number_topics, 'number_words:', number_words)

lda, count_vectorizer, number_words = learning_by_LDA(Data2,number_topics, number_words)
print('\n')


#Topic #1:
#the is and clean highly host very room come will
#
#Topic #2:
#good 設備齊全 place it love all definitely very stylish friendly
#
#Topic #3:
#the and nice to is place very stay for clean
#
#Topic #4:
#will thanks again come back so definitely great room home
#
#Topic #5:
#cozy and nice clean we time you great stay comfortable



#Topic #1:
#und wonderful gordon melissa in am super hotel so restaurants
#
#Topic #2:
#and to the it staying gordon is really nice hk
#
#Topic #3:
#and to the was gordon very is recommend station you
#
#Topic #4:
#easily fantastic hospitality great place and the gordon to hong
#
#Topic #5:
#and the to gordon is very was in of you



# =============================================================================
# Method 2: TF-IDF
# =============================================================================

import operator

from sklearn.feature_extraction.text import TfidfVectorizer

def print_top_tokens(feature_names, word_counts, n_top=10):
    tdidf_counts = zip(feature_names, word_counts.sum(axis=0).tolist()[0])
    sorted_x = sorted(dict(tdidf_counts).items(), key = operator.itemgetter(1), reverse = True)

    if n_top:
        return sorted_x[: n_top]
    else:
        return sorted_x

def learn_by_TF_IDF(Data, ngram_par0, ngram_parN):
    count_vectorizer = TfidfVectorizer(
                            strip_accents='unicode',
                            preprocessor=None,
                            analyzer='word',
                            ngram_range=(ngram_par0, ngram_parN),
                            min_df=10,
                            use_idf=True, smooth_idf=True, 
                            max_features = 5000
                        )
    
    bag_of_words = count_vectorizer.fit_transform(Data['comments'].values.astype('U'))
    
    #grouped_eng_reviews = Data.groupby('listing_id')
    #listing_2818 = Data.get_group(2818)
    
    # extract top relevant keywords
    listing_words = count_vectorizer.transform(Data['comments'])
    feature_names = count_vectorizer.get_feature_names()
    top_tokens = print_top_tokens(feature_names, listing_words)
#    print([token[0] for token in top_tokens])
    
    for token in top_tokens:
        print(token)
    return


ngram_par0 = 2
ngram_parN = 2

print('ngram_par0: ', ngram_par0, ' | ngram_parN: ', ngram_parN )
print('ID:8637229')
learn_by_TF_IDF(Data1,ngram_par0, ngram_parN)
print('ID:850531')
learn_by_TF_IDF(Data2,ngram_par0, ngram_parN)

print('\n')

ngram_par0 = 2
ngram_parN = 3

print('ngram_par0: ', ngram_par0, ' | ngram_parN: ', ngram_parN )
print('ID:8637229')
learn_by_TF_IDF(Data1,ngram_par0, ngram_parN)
print('ID:850531')
learn_by_TF_IDF(Data2,ngram_par0, ngram_parN)

print('\n')

ngram_par0 = 3
ngram_parN = 3

print('ngram_par0: ', ngram_par0, ' | ngram_parN: ', ngram_parN )
print('ID:8637229')
learn_by_TF_IDF(Data1,ngram_par0, ngram_parN)
print('ID:850531')
learn_by_TF_IDF(Data2,ngram_par0, ngram_parN)