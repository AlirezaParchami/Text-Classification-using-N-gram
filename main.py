import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def read_file():
    data = pd.read_csv('./Document/SMSSpamCollection', sep='\t')
    return data


def preprocessing():
    # Word Tokenization
    data['text'] = data['text'].apply(word_tokenize)
    # Remove Stopwords
    stopWords = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stopWords])
    # Stemming
    ps = PorterStemmer()
    data['text'] = data["text"].apply(lambda x: [ps.stem(y) for y in x])

def ngrams(n_g):
    # Generate n-grams
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(n_g,n_g), analyzer='word')
    dt_mat = cv.fit_transform(data['text'])
    # feature Generation. TF-IDF
    tfidf_transformer = TfidfTransformer()
    tfidf_mat = tfidf_transformer.fit_transform(dt_mat)
    return tfidf_mat


def train_test():
    # assign Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_mat, data['label'], test_size=0.2, random_state=1)
    # Train
    clf = MultinomialNB().fit(X_train, y_train)
    # Test
    predicted = clf.predict(X_test)
    print("accuracy :", metrics.accuracy_score(y_test, predicted))
    print("classification_report :\n", metrics.classification_report(y_test, predicted))
    print("confusion_matrix :\n", metrics.confusion_matrix(y_test, predicted,labels=["ham","spam"]))

data = read_file()
preprocessing()
print("-------------")
data['text'] = data['text'].apply(' '.join)
ngram_list = [1,2,3]
for i in ngram_list:
    print("n_gram = ",i)
    tfidf_mat = ngrams(i)
    train_test()
    print("---------------")
