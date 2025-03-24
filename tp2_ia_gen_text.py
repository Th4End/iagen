import pandas as pd
import re
from collections import defaultdict
import string
import random
from tensorflow.keras.preprocessing.text import Tokenizer



df = pd.read_excel("grimms_tale.xlsx")


def nettoyer_texte(texte):
    if isinstance(texte, str):
        texte = texte.lower()
        texte = texte.translate(str.maketrans("", "", string.punctuation))
    return texte


df["Story"] = df["Story"].apply(nettoyer_texte)


def table_transitions(texte):
    mots = texte.split()
    table = defaultdict(list)

    for i in range(len(mots)-1):
        table[mots[i]].append(mots[i+1])
    return dict(table)


tables_transitions = {}

for index, row in df.iterrows():
    if isinstance(row["Story"], str):
        tables_transitions[row["Title"]] = table_transitions(row["Story"])


def generer_phrase(tables_transitions, longueur_phrase=20):
    titre = random.choice(list(tables_transitions.keys()))
    transitions = tables_transitions[titre]


    mot_depart = random.choice(list(transitions.keys()))

    phrase = [mot_depart]


    while len(phrase) < longueur_phrase:
        mot_suivant = random.choice(transitions[phrase[-1]])
        phrase.append(mot_suivant)

    return ' '.join(phrase)


phrase_generee = generer_phrase(tables_transitions, 20)
print(phrase_generee)
