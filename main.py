import numpy as np
import nltk
import os
import email_read_util

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from nltk.metrics import edit_distance


def read_email_files():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = email_read_util.extract_email_text(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y


def read_email_files_load():
    X = []
    y = []
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = email_read_util.load(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y


def classify(X, y, clf, vectorizer):

    # Преобразование массива строк в структуру bag of words
    X_vector = vectorizer.fit_transform(X)

    # Обучение
    clf.fit(X_vector, y)

    # Оценка
    score = cross_val_score(clf, X_vector, y, cv=3)
    print('Accuracy: ', end='')
    print(score)
    print('Mean accuracy: ', end='')
    print(score.mean())


def compare(email_str0, email_str1, clf, vectorizer):

    # Классификация исходного письма
    Z = []
    Z.append(email_str0)
    Z_vector = vectorizer.transform(Z)
    label = clf.predict(Z_vector)[0]
    print('Predicted label: ', end='')
    print(label)

    # Классификация измененного письма
    Z = []
    Z.append(email_str1)
    Z_vector = vectorizer.transform(Z)
    label = clf.predict(Z_vector)[0]
    print('New label: ', end='')
    print(label)


nltk.download('punkt')
nltk.download('stopwords')

DATA_DIR = 'datasets/trec07p/data/'
LABELS_FILE = 'datasets/trec07p/full/index'

# Получаем метки классов
labels = {}
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0


print('Наивный Байес')
X, y = read_email_files()
vectorizer = CountVectorizer()
clf = MultinomialNB()
classify(X, y, clf, vectorizer)

# Сравнение измененного письма (extract_email_text)
print('Отравление Байеса')
filename = 'inmail.4'
email_str0 = email_read_util.extract_email_text(os.path.join(DATA_DIR, filename))
email_str1 = email_read_util.extract_email_text(os.path.join(filename))
ind = X.index(email_str0)
print('First label: ', end='')
print(y[ind])
print('Edit distance: ', end='')
print(edit_distance(email_str0, email_str1))

compare(email_str0, email_str1, clf, vectorizer)

print('Замена extract_email_text на load')
X, y = read_email_files_load()
classify(X, y, clf, vectorizer)
email_str0 = email_read_util.load(os.path.join(DATA_DIR, filename))
email_str1 = email_read_util.load(os.path.join(filename))
compare(email_str0, email_str1, clf, vectorizer)

print('Биграммы')
vectorizer = CountVectorizer(ngram_range=(2, 2))
classify(X, y, clf, vectorizer)
compare(email_str0, email_str1, clf, vectorizer)

print('TF/IDF')
vectorizer = TfidfVectorizer()
classify(X, y, clf, vectorizer)
compare(email_str0, email_str1, clf, vectorizer)

print('Случайный лес')
clf = RandomForestClassifier()
classify(X, y, clf, vectorizer)
compare(email_str0, email_str1, clf, vectorizer)
