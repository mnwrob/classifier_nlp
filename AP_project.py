import numpy as np
import re
import nltk
from sklearn import datasets
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

print("load start")
dataset = load_files(r"/Users/marion/school/classifier_nlp/data", encoding="latin-1")
print("load complete")

print("X,y start")
X, y = dataset.data, dataset.target
print("X,Y complete")


print("len(X) = ", len(X))

X = X[:2000]
y = y[:2000]

print("len(X) = ", len(X))


from nltk.tokenize import word_tokenize

scripts = []

for sen in range(0, len(X)):
    # remove all new lines
    script = re.sub(r'\\n', ' ', str(X[sen]))

    # convert to lowercase
    script = script.lower()

    # remove all the special characters
    script = re.sub(r'[^a-z\d\']+', ' ', script)

    # substitute multiple spaces with single space
    script = re.sub(r' +', ' ', script)

    # tokenization
    script = nltk.word_tokenize(script)

    # join and append data
    script = ' '.join(script)
    
    scripts.append(scripts)


from nltk.corpus import wordnet

# translate nltk tags to wordnet tags
def nltk_to_wordnet(nltk_tag):
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


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    # tokenize sentence, find the POS tags 
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    # create tuple of token and wordnet_tag
    wordnet_tagged = map(lambda x: (x[0], nltk_to_wordnet(x[1])), nltk_tagged)
    lemmatized_sen = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if no tag, append the token without one
            lemmatized_sen.append(word)
        else:        
            # else use tag to lemmatize token
            lemmatized_sen.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sen)

print("finished lemmatizing")


from sklearn.feature_extraction.text import CountVectorizer

# extract features and convert to numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(scripts).toarray()


from sklearn.model_selection import train_test_split

# divide data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier

# train model using fit and RandomForestClassifier method 
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

# predict test data using predict method
y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# evaluate train model using sklearn.metrics
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
