# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:33:31 2021

@author: shrey
"""

import pandas as pd
import numpy as np
import re
import enchant
d = enchant.Dict("en_US")
from contractions import contractions

df = pd.read_csv("fable_data.csv")

def clean(text):
    text = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
    text = re.sub('[0-9]+[,\.][0-9]+','', text)
    text = re.sub('([a-z]+)?[0-9]+([a-z]+)?','', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub("([?.!,])", r" \1 ", text)
    new_text = []
    word = (text.lower())
    for w in word.split():
        if w in contractions.keys():
            new_text.append(contractions[w])
        else:
            new_text.append(w)
    text = ' '.join(new_text)
    text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)
    return text

def remove_not_words(text):
    not_words = []
    sentence = text.split()
    for word in sentence:
        if (len(word) == 1 and word not in ['i', 'a', '.', '?', '!', ',']):
            not_words.append(word)
        if (len(word) == 2 or len(word) == 3):
            if d.check(word) == False:
                not_words.append(word)
    [sentence.remove(i) for i in not_words]
    text = " ".join(sentence)
    return text

text = df['Reviews'].map(clean)
text = text.map(remove_not_words)

f = open("reviews.txt", "w")
for lines in text:
    f.write(lines)
    f.write("\n")
f.close()
