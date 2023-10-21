import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import ssl
import openpyxl
import nltk

nltk.download('punkt')

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import *
from nltk import word_tokenize

ssl._create_default_https_context = ssl._create_unverified_context

cats = ['comp.windows.x', 'rec.sport.baseball', 'rec.sport.hockey']
rm = ('headers', 'footers', 'quotes')

parameters = {
    'vect__max_features': (100, 500, 1000, 5000, 10000),
    'vect__stop_words': ('english', None),
    'tfidf__use_idf': (True, False),
}

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])


def get_data(categories, remove):
    twenty_train_full = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
    twenty_test_full = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

    twenty_train_full = twenty_train_full.data
    twenty_test_full = twenty_test_full.data

    twenty_train = dict()
    twenty_test = dict()

    for category in categories:
        twenty_train[category] = fetch_20newsgroups(subset='train', categories=[category], remove=remove)
        twenty_test[category] = fetch_20newsgroups(subset='test', categories=[category], remove=remove)
        twenty_train[category] = twenty_train[category].data
        twenty_test[category] = twenty_test[category].data

    twenty_train['full'] = twenty_train_full
    twenty_test['full'] = twenty_test_full
    print('Done!')

    return twenty_train, twenty_train_full, twenty_test, twenty_test_full


def stemming(data):
    porter_stemmer = PorterStemmer()
    stem = []
    for text in data:
        nltk_tokens = word_tokenize(text)
        line = ''
        for word in nltk_tokens:
            line += ' ' + porter_stemmer.stem(word)
            stem.append(line)
    return stem


def sort_by_tf(input_str):
    return input_str[1]


def top_list(vect, data, count):
    x = list(zip(vect.get_feature_names_out(), np.ravel(data.sum(axis=0))))
    x.sort(key=sort_by_tf, reverse=True)
    return x[:count]


def fill_frame(frame):
    pass


def process(train, categories):
    cats = categories[:]
    cats.append('full')

    mux = pd.MultiIndex.from_product([['Count', 'TF', 'TF-IDF'], ['Без стоп-слов', 'Со стоп-словами']])
    summary = dict()

    for category in cats:
        summary[category] = pd.DataFrame(columns=mux)

    stop_words = [None, 'english']
    idf = [False, True]

    index_stop = {
        'english': 'Без стоп-слов',
        None: 'Со стоп-cловами'
    }

    index_tf = {
        False: 'TF',
        True: 'TF-IDF'
    }

    for category in cats:
        for stop in stop_words:
            vect = CountVectorizer(max_features=10000, stop_words=stop)
            vect.fit(train[category])
            train_data = vect.transform(train[category])
            summary[category]['Count', index_stop[stop]] = top_list(vect, train_data, 20)

            for tf in idf:
                tfidf = TfidfTransformer(use_idf=tf).fit(train_data)
                train_fidf = tfidf.transform(train_data)
                summary[category][index_tf[tf], index_stop[stop]] = top_list(vect, train_fidf, 20)

    return summary


def print_classification_score(clf, data):
    print(classification_report(clf.predict(data.data), data.target))


def prespocess(data, max_features, stop_words, use_tf, use_idf):
    tf = None
    cv = CountVectorizer(max_features=max_features, stop_words=stop_words).fit(data)
    if use_tf:
        tf = TfidfTransformer(use_idf=use_idf).fit(cv.transform(data))
    return cv, tf


def models_grid_search(data_train, data_test):
    max_features = [100, 500, 1000, 5000, 10000]
    stop_words = ['english', None]
    use_tf = [True, False]
    use_idf = [True, False]
    res = dict()
    for param in itertools.product(max_features, stop_words, use_tf, use_idf):
        cv, tf = prespocess(data_train.data, param[0], param[1], param[2], param[3])
        if tf:
            clf = MultinomialNB().fit(tf.transform(cv.transform(data_train.data)), data_train.target)
            prep_test = tf.transform(cv.transform(data_test.data))
        else:
            clf = MultinomialNB().fit(cv.transform(data_train.data), data_train.target)
            prep_test = cv.transform(data_test.data)

        name = f'max_features={param[0]}_stop_words={param[1]}_use_tf={param[2]}_use_idf ={param[3]}'
        res[name] = pd.DataFrame(classification_report(clf.predict(prep_test), data_test.target, output_dict=True))
    return res


twenty_train, twenty_train_full, twenty_test, twenty_test_full = get_data(cats, rm)

stem_train = dict()
stem_test = dict()

for category in cats:
    stem_train[category] = stemming(twenty_train[category])
    stem_test[category] = stemming(twenty_test[category])

stem_train['full'] = stemming(twenty_train['full'])
stem_test['full'] = stemming(twenty_test['full'])

summ_without_stem = process(twenty_train, cats)
summ_with_stem = process(stem_train, cats)

if not os.path.exists('summ_without_stem'):
    os.makedirs('summ_without_stem')

if not os.path.exists('summ_with_stem'):
    os.makedirs('summ_with_stem')

for cat in ['full'] + cats:
    summ_without_stem[cat].to_excel('summ_without_stem/' + cat + '.xlsx')
    summ_with_stem[cat].to_excel('summ_with_stem/' + cat + '.xlsx')

print('Well Done!')

# Pipelines

twenty_train_full = fetch_20newsgroups(subset='train', categories=cats, remove=rm)
twenty_test_full = fetch_20newsgroups(subset='test', categories=cats, remove=rm)

scores = models_grid_search(twenty_train_full, twenty_test_full)
if not os.path.exists('scores'):
    os.makedirs('scores')

print(len(scores.items()))

for name, score in scores.items():
    print(name)
    score.to_excel('scores/' + name + '.xlsx')

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=3)
gs_clf.fit(X=twenty_train_full.data, y=twenty_train_full.target)
print_classification_score(gs_clf, twenty_test_full)
