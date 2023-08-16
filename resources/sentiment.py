""" SENTIMENT.PY

Includes solutions for the sentiment analysis assignments (Section 8).
"""

from typing import List, Dict, Tuple
from functools import reduce
from utils import to_word_list
import numpy as np


from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers, Model

from utils import train_model


titles = [
    "פסק-דין", "פסק דין", "החלטה",
    "פסק-דין חלקי", "פסק-דין (חלקי)", "פסק דין (חלקי)",
    "פסק-דין משלים", "פסק-דין (משלים)", "פסק דין (משלים)",
    "החלטה (בעניין המשיב 1) ופסק-דין (בעניין המשיב 2)"
]

pos_words = [
    "מתקבל", "מתקבלת", "מתקבלים", "מתקבלות", "מקבל",
    "מקבלת", "מקבלים", "מקבלות", "קבלת", "קבלה"
]

neg_words = [
    "נדחה", "נדחית", "נדחים", "נדחות", "דוחה", "דוחים",
    "דוחות", "דחיית", "דחייה"
]

def sentiment_tag(raw_docs: Dict[str, List[str]]) -> Dict[str, Tuple[List[str], str]]:
    """Prepares the data by trimming anything before the actual content of the verdict and
    tagging each document based on `pos_words` and `neg_words` (rule-based classification).
    """

    doc_contents = {}

    for dname, doc in raw_docs.items():
        for t in titles:
            if t in doc:
                t_ind = doc.index(t)
        
        doc_contents[dname] = doc[t_ind + 1:]

    
    doc_tags = {}

    for dname, dcontent in doc_contents.items():
        dcontent = to_word_list(" ".join(dcontent))

        neg_count = pos_count = 0

        for w in dcontent:
            if w in pos_words:
                pos_count += 1
            elif w in neg_words:
                neg_count += 1

        if pos_count > neg_count:
            doc_tags[dname] = "POSITIVE"
        elif pos_count < neg_count:
            doc_tags[dname] = "NEGATIVE"
        else:
            doc_tags[dname] = "NEUTRAL"

    return doc_tags


def prep_data(docs: Dict[str, List[str]], targets: Dict[str, str]):
    vocab = set(reduce(lambda x,y: x+y, docs.values()))
    word2idx = {w: i for i,w in zip(range(1, len(vocab)+1), vocab)}
    tag2idx = {t: i for i,t in enumerate(["POSITIVE", "NEGATIVE", "NEUTRAL"])}

    doc_names = list(docs.keys())
    data_seqs = [[word2idx[w] for w in docs[dname]] for dname in doc_names]
    data_seqs = pad_sequences(sequences=data_seqs, value=0)
    tags = np.array([tag2idx[targets[dname]] for dname in doc_names])
    
    t_data, v_data, t_tags, v_tags, t_docs, v_docs = train_test_split(
        data_seqs, tags, doc_names,
        test_size=0.2,
        random_state=1
    )

    return t_data, v_data, t_tags, v_tags, t_docs, v_docs, word2idx, tag2idx

def build_model(vocab_size, num_tags, l1_size=2000, l2_size: int=100):
    inp = layers.Input((None,))
    out1 = layers.Embedding(input_dim=vocab_size + 1, output_dim=l1_size)(inp)
    out2 = layers.GRU(l2_size, return_sequences=False)(out1)
    out = layers.Dense(num_tags, activation="softmax")(out2)

    model = Model(inp, out)

    return model

def sentiment_model_predictions(docs: Dict[str, List[str]], targets: Dict[str, str]):
    t_data, v_data, t_tags, v_tags, t_docs, v_docs, word2idx, tag2idx = prep_data(docs, targets)
    
    model = build_model(len(word2idx), len(tag2idx))
    train_model(model, t_data, v_data, t_tags, v_tags, epochs=5, batch_size=1)

    idx2word = {i: w for w,i in word2idx.items()}
    idx2word[len(idx2word)] = "<PAD>"
    idx2tag = {i: t for t,i in tag2idx.items()}

    data = np.concatenate((t_data, v_data))
    tags = np.concatenate((t_tags, v_tags))
    doc_names = t_docs + v_docs
    results = {}

    for dname, d_seq, tag in tqdm(zip(doc_names, data, tags), desc="Generating Predictions . . ."):
        pred = np.argmax(model.predict(np.array([d_seq]), verbose=0)[0]).item()
        results[dname] = idx2tag[pred]
    
    return results