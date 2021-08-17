import numpy as np
import re
import nltk
from sklearn import datasets
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

dataset = load_files(r"/Users/marion/school/classifier_nlp/data")
X, y = dataset.data, dataset.target

documents = []

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stem = WordNetLemmatizer()

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

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





