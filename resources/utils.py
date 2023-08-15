""" UTILS.PY

Includes necessary utilites for the project.
"""

from typing import List
import string

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration


### HTML FILE TO RAW TEXT ###

def tag_visible(element) -> bool:
    """Helper method. Determines whether a tag is indeed visible on the page.

    From: https://stackoverflow.com/a/1983219 , modified a bit.
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment) or isinstance(element, Declaration):
        return False
    return True

def scrape_html(filepath: str) -> List[str]:
    """Scrapes visible text from html (As it is the easiest and sanest approach by far).

    Uses helper method `tag_visible` to determine whether the text is visible in the page or not.
    """
    with open(filepath, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")
        back = []
        
        texts = soup.findAll(string=True)
        for text in texts:
            if tag_visible(text) and not (text == "" or text.isspace()):
                back.append(text)
    
    return back


### RAW TEXT TO WORD LIST ###

def to_word_list(raw_text: str) -> List[str]:
    """Translates the output of the `scrape_html` method to a list of words.
    
    Also cleans the output by:
    - Concatenating single letters in some cases.
    - Removing punctuation.
    - Removing words which do not contain hebrew letters.
    """
    text = raw_text.replace(u"נ ג ד", u"נגד")
    text = text.replace(u"ש ו פ ט ת", u"שופטת")
    text = text.replace(u"ש ו פ ט", u"שופט")
    text = text.replace(u"ה נ ש י א ה", u"הנשיאה")
    text = text.replace(u"ר ש מ ת", u"רשמת")

    words = text.split()
    punct = tuple(string.punctuation)

    clean_words = []

    for i in range(len(words)):
        word = words[i]

        while word.startswith(punct):
            word = word[1:]
        
        while word.endswith(punct):
            word = word[:-1]
        
        if word != "" and any((c in word) for c in "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"):
            clean_words.append(word)

    return clean_words