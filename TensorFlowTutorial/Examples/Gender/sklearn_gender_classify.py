"""Using scikit-learn for gender classification by full names
"""

import os 
import sys
from collections import Counter
import numpy as np
np.random.seed(1337)  # for reproducibility

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    names   = []
    genders = []
    with open("students.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            name, gender = line.split(",")
            name   = name.strip()
            gender = gender.strip()
            names.append(name)
            genders.append(gender)

    print("Number of names: %d" % len(names))
    print("Label distribution")
    print(Counter(genders))
    print()
    
    print("Training model")
    vectorizer = CountVectorizer(binary=True,
                                 ngram_range=(1,2))
    clf = MultinomialNB(alpha=.01)
    pip = Pipeline([
                ('vect', vectorizer),
                ('classification', clf),
                ])
    N = 10
    scores = cross_val_score(pip, names, genders, cv=N,
                             scoring = 'accuracy')
    print(scores)
    score = scores.mean()
    print("average accuracy: %0.3f" % scores.mean())
    print()





