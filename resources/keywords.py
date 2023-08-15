""" KEYWORDS.PY

Includes solutions for the keyword extraction assignments (Sections 3, 4, 10).
"""

from typing import Dict, List
import numpy as np
from collections import OrderedDict
from functools import reduce

from gensim.models import Word2Vec


from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers, Model

from utils import train_model

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

def prep_data(docs: Dict[str, List[str]]):
    vocab = set(reduce(lambda x,y: x+y, docs.values()))
    word2ind = {w: i for i,w in zip(range(1, len(vocab)+1), vocab)}
    sequences = [[word2ind[w] for w in d] for d in docs.values()]

    split_seqs = [[s[i:i+50] for i in range(0, len(s), 50)] for s in sequences]
    split_seqs = reduce(lambda x,y: x+y, split_seqs)
    data = pad_sequences(sequences=split_seqs, value=0)
    
    t_data, v_data = train_test_split(data, test_size=0.2, random_state=1)

    return t_data, v_data, word2ind


def build_model(vocab_size, h_size=2000, emb_size: int=400):
    inp = layers.Input((None,))
    emb = layers.Embedding(input_dim=vocab_size + 1, output_dim=h_size)(inp)
    enc_out = layers.GRU(emb_size, return_sequences=True)(emb)

    dec_in = layers.GRU(emb_size, return_sequences=True)(enc_out)
    dec_h = layers.Dense(h_size)(dec_in)
    dec_out = layers.Dense(vocab_size + 1, activation="softmax")(dec_h)

    autoencoder = Model(inp, dec_out)
    encoder = Model(inp, enc_out)

    return autoencoder, encoder


def get_autoencoder_scores(
        docs: Dict[str, List[str]],
        epochs: int=15,
        emb_size: int=400
    ) -> Dict[str, Dict[str, float]]:
    
    """Generates a score for each unique word in each document using the autoencoder method to generate
    embeddings and then use those in `get_scores_by_embeddings`.
    """

    t_data, v_data, word2ind = prep_data(docs)
    autoencoder, encoder = build_model(len(word2ind), emb_size=emb_size)
    train_model(autoencoder, t_data, v_data, t_data, v_data, epochs=epochs)

    # generate embeddings
    data = np.concatenate((t_data, v_data))
    ind2word = {i: w for w,i in word2ind.items()}
    word_freqs, emb_sums = {}, {}
    sliced_data = [data[i:i+32] for i in range(0, len(data), 32)]

    for sd in tqdm(sliced_data, desc="Generating Embeddings . . ."):
        spred = encoder.predict(np.array(sd), verbose=0)[0]

        for d, pred in zip(sd, spred):
            for w_ind, embs in zip(d, pred):
                if w_ind != 0:
                    word = ind2word[w_ind]
                    word_freqs[word] = word_freqs.get(word, 0) + 1
                    emb_sums[word] = emb_sums.get(word, np.zeros((emb_size,))) + embs

    embeddings = {}

    for word in emb_sums.keys():
        embeddings[word] = emb_sums[word] / word_freqs[word]
    
    return get_scores_by_embeddings(docs, embeddings)



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
    elif mode in ["autoencoder", "auto encoder", "auto-encoder", "ae"]:
        scores = get_autoencoder_scores(docs)
    else:
        raise ValueError(f"argument 'mode' must be one of 'tf-idf', 'word2vec' or 'autoencoder'")

    # Sort and return scores
    for doc_name, doc_scores in scores.items():
        sorted_values = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
        if num_top_keywords is not None:
            sorted_values = sorted_values[:num_top_keywords]
        
        scores[doc_name] = OrderedDict(sorted_values)
    
    return scores