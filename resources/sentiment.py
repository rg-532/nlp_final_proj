""" SENTIMENT.PY

Includes solutions for the sentiment analysis assignments (Section 8).
"""

from typing import List, Dict, Tuple
from utils import to_word_list


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