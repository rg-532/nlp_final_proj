""" NER.PY

Includes solutions for the named entity recognition assignments (Sections 2, 6).
"""

import re
from typing import List, Dict


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

