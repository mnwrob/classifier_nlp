import numpy as np
import re
import nltk
from sklearn import datasets
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

print("load start")
dataset = load_files(r"/Users/marion/school/classifier_nlp/data")
print("load complete")

print("X,y start")
X, y = dataset.data, dataset.target
print("X,Y complete")

documents = []

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stem = WordNetLemmatizer()

print("len(X) = ", len(X))

X = X[:1000]
y = y[:1000]

print("len(X) = ", len(X))

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'/[^a-z\d ]+/i', ' ', str(X[sen]))

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', str(X[sen]), flags=re.I)

    # Removing all new lines
    document = re.sub(r'\\n', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Removing prefixed 'b'
    # document = re.sub(r'^b\s+', '', document)

    # Lemmatization
    document = document.split()

    document = [stem.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

print("finished lemmatizing")

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))




