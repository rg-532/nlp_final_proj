""" KEYWORDS.PY

Includes solutions for the keyword extraction assignments (Sections 3, 4, 10).
"""

from typing import Dict, List
import numpy as np
from collections import OrderedDict
from functools import reduce

from gensim.models import Word2Vec

import torch
import torch.nn as nn


### SECTION 3 ###

def get_tf_idf_scores(docs: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:

    """Generates a score for each unique word in each document using the term frequency-inverse document
    frequency (tf-idf) algorithm.
    """
    tf_scores = {}
    df_scores = {}

    for doc_name, doc in docs.items():
        tf_scores[doc_name] = {}
        
        for word in doc:
            tf_scores[doc_name][word] = tf_scores[doc_name].get(word, 0) + 1
        
        for word in tf_scores[doc_name].keys():
            df_scores[word] = df_scores.get(word, 0) + 1
    
    tf_idf_scores = {}

    for doc_name, doc_tfs in tf_scores.items():
        tf_idf_scores[doc_name] = {}

        for word, word_tf in doc_tfs.items():
            tf_idf_scores[doc_name][word] = word_tf * np.log(len(docs) / df_scores[word])
    
    return tf_idf_scores


### SECTION 4 ###

def get_scores_by_embeddings(
        docs: Dict[str, List[str]],
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:

    """Helper method. Uses given embeddings to generate a score for each unique word in each document.
    
    We first define proximity between two words to be the result of the Gaussian Similarity Function.
    Due to high memory consumption and computation time, instead of scoring words by their average
    proximity to all words in a document for each document, we compute the embeddings of an entire
    document by computing the average of the embeddings of the words within the document, weighed by
    their relative frequency.
    """
    doc2ind = {d: i for d, i in zip(docs.keys(), range(len(docs)))}
    word2ind = {w: i for w, i in zip(embeddings.keys(), range(len(embeddings)))}
    ind2word = {i: w for w, i in word2ind.items()}

    word_rel_freqs = np.zeros((len(docs), len(embeddings)), dtype=float)

    for doc_name, doc in docs.items():
        doc_size = len(doc)

        for word in doc:
            word_rel_freqs[doc2ind[doc_name], word2ind[word]] += 1 / doc_size

    word_embs_mat = np.stack([embeddings[ind2word[i]] for i in range(len(embeddings))])
    doc_embs_mat = np.matmul(word_rel_freqs, word_embs_mat)
    
    emb_scores = {}
    
    for doc_name in docs.keys():
        emb_scores[doc_name] = {}

        for word in embeddings.keys():
            emb_scores[doc_name][word] = np.exp(-np.sum(
                (doc_embs_mat[doc2ind[doc_name]] - word_embs_mat[word2ind[word]])**2
            ))
    
    return emb_scores


def get_word2vec_scores(
        docs: Dict[str, List[str]],
        epochs: int=30,
        emb_size: int=400
    ) -> Dict[str, Dict[str, float]]:
    
    """Generates a score for each unique word in each document using the word2vec method to generate
    embeddings and then use those in `get_scores_by_embeddings`.
    """
    model = Word2Vec(sentences=docs.values(), epochs=epochs, vector_size=emb_size, min_count=0)
    embeddings = {w: model.wv[w] for w in model.wv.key_to_index}

    return get_scores_by_embeddings(docs, embeddings)


### SECTION 10 ###




# General method for external use:
def extract_keywords(
        docs: Dict[str, List[str]],
        mode: str,
        num_top_keywords: int | None = 5
    ) -> Dict[str, OrderedDict[str, float]]:

    """Given a set of documents as word lists, calculates a score for every word in each document.
    The score meaures the degree to which the word fits to serve as a keyword of the document.
    These scores are then sorted from high to low.

    Parameters
    ----------

    `docs` : Dict[str, List[str]]
            Dictionary of documents by filenames, with the documents translated to word lists.
    
    `mode` : {'tf-idf', 'word2vec', 'autoencoder'}
            Technique to use for generating scores.
    
    `num_top_keywords` : int, default=5
            Number of desired best-fitting keywords. Must be positive if specified.

    Returns
    -------

    Dict[str, OrderedDict[str, float]]
            Sorted pairs of words and their scores for each document.
    """

    # Catch errors with `num_top_keywords`
    if num_top_keywords is not None:
        if not isinstance(num_top_keywords, int):
            int_class = int.__name__
            this_class = num_top_keywords.__class__.__name__
            raise TypeError(f"argument 'num_top_keywords' must be of type '{int_class}', " +
                            f"instead got argument of type '{this_class}'")
        elif num_top_keywords <= 0:
            raise ValueError(f"argument 'num_top_keywords' must be positive, got {num_top_keywords}")

    # Get scores based on `mode`
    mode = mode.lower().strip()

    if mode in ["tf-idf", "tf idf", "tfidf"]:
        scores = get_tf_idf_scores(docs)
    elif mode in ["word2vec", "word 2 vec", "w2v", "wordtovec", "word to vec", "wtv"]:
        scores = get_word2vec_scores(docs)
    else:
        raise ValueError(f"argument 'mode' must be one of 'tf-idf', 'word2vec' or 'autoencoder'")

    # Sort and return scores
    for doc_name, doc_scores in scores.items():
        sorted_values = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
        if num_top_keywords is not None:
            sorted_values = sorted_values[:num_top_keywords]
        
        scores[doc_name] = OrderedDict(sorted_values)
    
    return scores