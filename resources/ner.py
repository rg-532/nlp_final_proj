""" NER.PY

Includes solutions for the named entity recognition assignments (Sections 2, 6).
"""

import re
from typing import List, Dict
import numpy as np


from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM,Embedding
from tensorflow.keras.layers import InputLayer, SpatialDropout1D, Bidirectional

from tqdm import tqdm

from utils import train_model


### SECTION 2 ###

def get_judges(parsed_sentences, parsed_tags, html_txt):
    judges = []
    seen_judge = False

    for index,txt in enumerate(html_txt):
        if seen_judge:
            if re.search(":$",txt):
                parsed_sentences.append(txt.split(" "))
                parsed_tags.append(['O'] * len(txt.split(" ")))
                return judges, index+1

        if re.search("^כבוד",txt):
            seen_judge = True
            parsed_sentences.append([txt])
            judges.append(txt)
            parsed_tags.append(['Jugde'])
        else:
            parsed_sentences.append(txt.split(" "))
            parsed_tags.append(['O'] * len(txt.split(" ")))


def get_prosecutors(parsed_sentences, parsed_tags, index, html_txt):
    prosecutors = []

    while(html_txt[index]!='נ' and html_txt[index]!='נ ג ד') :
        prosecutor = ''.join([i for i in html_txt[index] if not i.isdigit() and i!='.'])
        prosecutors.append(prosecutor)
        parsed_sentences.append([prosecutor])
        parsed_tags.append(['Prosecutor'])
        index += 1
    
    parsed_sentences.append([html_txt[index]])
    parsed_tags.append(['O'])

    return prosecutors,index + 4


def get_defendants(parsed_sentences, parsed_tags, index, html_txt):
    defendants = []

    if not re.search("^[1-9].",html_txt[index]):
        defendant = ''.join([i for i in html_txt[index] if not i.isdigit() and i!='.'])
        defendants.append(defendant)
        parsed_sentences.append([defendant])
        parsed_tags.append(['Defendant'])
        return defendants,index+1

    while(re.search("^[1-9].",html_txt[index])):
        defendant = ''.join([i for i in html_txt[index] if not i.isdigit() and i!='.'])
        defendants.append(defendant)
        parsed_sentences.append([defendant])
        parsed_tags.append(['Defendant'])
        index +=1

    return defendants, index

def get_date(parsed_sentences,parsed_tags,index,html_txt):
    while(not re.search("\u200f[0-9]+[.][0-9]+[.][0-9][0-9]+",html_txt[index]) and ('תאריך הישיבה:' not in html_txt[index]) and ('ניתן היום' not in html_txt[index])):
        parsed_sentences.append(html_txt[index].split(" "))
        parsed_tags.append(['O'] * len(html_txt[index].split(" ")))
        index += 1
        if index == len(html_txt):
            return None
        
    if re.search("\u200f[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",html_txt[index]):
        string = re.search("\u200f[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",html_txt[index])

        if string:
            parsed_sentences.append([string])
            parsed_tags.append(['Date'])
            return string[0].replace("\u200f","")
        else:
            return None
    
    elif 'תאריך הישיבה:' in html_txt[index] or 'ניתן היום' in html_txt[index]:
        while(not re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",html_txt[index])):
            parsed_sentences.append(html_txt[index].split(" "))
            parsed_tags.append(['O']*len(html_txt[index].split(" ")))
            index += 1
            if index==len(html_txt):
                return None

        parsed_sentences.append([re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",html_txt[index])[0]])
        parsed_tags.append(['Date'])
        return re.search("[1-9][0-9]*[.][1-9][0-9]*[.][0-9][0-9]+",html_txt[index])[0]
    
    else:
        return None


def extract_ner(raw_docs: Dict[str, List[str]]):
    doc_names = []
    judges = []
    prosecutors = []
    defendants = []
    dates = []
    parsed_sentences = []
    parsed_tags = []

    for dname, doc in raw_docs.items():
        doc_names.append(dname)

        judges_index = get_judges(parsed_sentences, parsed_tags, doc)
        judges.append(judges_index[0])

        prosecutors_index = get_prosecutors(parsed_sentences, parsed_tags, judges_index[1], doc)
        prosecutors.append(prosecutors_index[0])

        defendants_index = get_defendants(parsed_sentences, parsed_tags, prosecutors_index[1], doc)
        defendants.append(defendants_index[0])

        date = get_date(parsed_sentences, parsed_tags, defendants_index[1], doc)
        dates.append(date)
    
    return {
        "Document": doc_names, 
        "Judges": judges,
        "Prosecutors": prosecutors,
        "Defendants": defendants,
        "Date": dates
    }, {
        "Parsed_Sentences": parsed_sentences,
        "Parsed_Tags": parsed_tags
    }


### SECTION 6 ###

def prep_data(tagged_sents):
    words = [item for sublist in tagged_sents["Parsed_Sentences"] for item in sublist]
    tags = [item for sublist in tagged_sents["Parsed_Tags"] for item in sublist]
    
    word2idx = {w: i for i,w in enumerate(set(words))}
    tag2idx = {t: i for i,t in enumerate(set(tags))}

    data_seqs = [[word2idx[w] for w in words[i:i+50]] for i in range(0, len(words), 50)]
    data_seqs = pad_sequences(sequences=data_seqs, padding="post", value=len(word2idx))

    tag_seqs = [[tag2idx[t] for t in tags[i:i+50]] for i in range(0, len(tags), 50)]
    tag_seqs = pad_sequences(sequences=tag_seqs, padding="post", value=tag2idx["O"])
    
    t_data, v_data, t_tags, v_tags = train_test_split(
        data_seqs, tag_seqs,
        test_size=0.2,
        random_state=1
    )

    return t_data, v_data, t_tags, v_tags, word2idx, tag2idx

def build_model(in_size, vocab_size, num_tags):
    model = keras.Sequential()
    model.add(InputLayer((in_size)))
    model.add(Embedding(input_dim=vocab_size+1, output_dim=in_size, input_length=in_size))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
    model.add(keras.layers.Dense(num_tags, activation="softmax"))

    return model

def ner_model_predictions(tagged_sents):
    t_data, v_data, t_tags, v_tags, word2idx, tag2idx = prep_data(tagged_sents)
    
    max_len = max([len(d) for d in t_data])
    
    model = build_model(max_len, len(word2idx), len(tag2idx))
    train_model(model, t_data, v_data, t_tags, v_tags)

    idx2word = {i: w for w,i in word2idx.items()}
    idx2word[len(idx2word)] = "<PAD>"
    idx2tag = {i: t for t,i in tag2idx.items()}

    data = np.concatenate((t_data, v_data))
    tags = np.concatenate((t_tags, v_tags))
    results = []

    for d_seq, t_seq in tqdm(zip(data, tags), desc="Generating Predictions . . ."):
        pred = np.argmax(model.predict(np.array([d_seq]), verbose=0)[0], axis=-1)
        results.append([(idx2word[wi], idx2tag[ti], idx2tag[pi])
                        for wi,ti,pi in zip(d_seq, t_seq, pred)])
    
    return results