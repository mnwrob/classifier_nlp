import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn import datasets
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Loading data
print("load start")
dataset = load_files(r"/Users/marion/school/classifier_nlp/data", encoding="latin-1")
print("load complete")

print("X,y start")
X, y = dataset.data, dataset.target
print("X,Y complete")

# Optional:
# Adjusting size of data
print("len(X) = ", len(X))
#X = X[:1000]
#y = y[:1000]
print("len(X) = ", len(X))

# Text preprocessing
documents = []
for sen in range(0, len(X)):
    # Removing all new lines
    document = re.sub(r'\\n', ' ', str(X[sen]))

    # Converting to Lowercase
    document = document.lower()

    # Removing all special characters
    document = re.sub(r'[^a-z\d\']+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r' +', ' ', document)
    
    documents.append(document)

# Needed for correct form of lemmatization:
# Translating nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# Lemmatizing with correct POS tags
lemmatizer = WordNetLemmatizer()
def lemmatize_sentence(sentence):
    # Tokenizing sentence and finding POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    # Creating tuple of token and wordnet tag
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    # Lemmatizing according to tag
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # If there is no available tag, appending token without one
            lemmatized_sentence.append(word)
        else:        
            # Else using tag to lemmatize token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

lemmatized_documents =[]
for document in documents:
    lemmatized_documents.append(lemmatize_sentence(document))

print("finished lemmatizing")


# Vectorizing data
vectorizer = CountVectorizer(max_df=0.4)
X = vectorizer.fit_transform(lemmatized_documents).toarray()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training data with random forest classifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

# Using classifier to predict correct label of test data
y_pred = classifier.predict(X_test)

# Printing some metrics about the model's performance
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
