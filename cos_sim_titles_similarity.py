import pandas as pd
import numpy as np
import sys
import datetime
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(df):
    '''
        Does:
            Uses tf idf and cosine sim to find similarities between the titles bu using their skills
            then saves Similarity(cos_sim) matrix
        Parametres:
            tfidfvectorizer: instance of the TF IDF
            sparse_matrix: calculates of accurence of each skill in titles and converts into matrix
            df_tf_idf: is DataFrame which shows accurence of skill
            cos_sim: is numpy array which applies df_tf_idf in cartesien format and shows each titles similarities
        return:
    '''

    tfidfvectorizer = TfidfVectorizer()
    # Create sparse matrix using the column needed
    sparse_matrix = tfidfvectorizer.fit_transform(df['skill'])
    # Sparse matrix is mostly zero but dense matrix is not (we have to use dense for this situation)
    doc_term_matrix = sparse_matrix.todense()
    # Creating df with dense matrix
    df_tf_idf = pd.DataFrame(doc_term_matrix,
                      columns=tfidfvectorizer.get_feature_names())
    # Applying cos-sim to our dense df
    cos_sim_matrix=cosine_similarity(df_tf_idf , df_tf_idf )
    # Saving the created matrix
    with open('cos_sim_title_similarity_matrix.npy', 'wb') as f:
        np.save(f, cos_sim_matrix)

if __name__ == '__main__':
    # Read the data
    df=pd.read_pickle('data.pkl')
    # Call the cos_sim func
    cos_sim(df)
