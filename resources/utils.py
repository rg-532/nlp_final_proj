import os
import sys

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration


### GETTING DATA FROM FILES ###

def tag_visible(element):
    """Helper method. Determines whether a tag is indeed visible on the page.

    From: https://stackoverflow.com/a/1983219 , modified a bit.
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment) or isinstance(element, Declaration):
        return False
    return True

def scrape_html(filepath: str):
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