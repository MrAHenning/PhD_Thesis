import pandas as pd
import nltk
import sklearn as sk
import string
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from joblib import dump, load
import numpy as np

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


def pre_process(s):
    # remove punctuation
    s = "".join([char for char in s if char not in string.punctuation])

    # filter stopwords
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    s_tokenized = nltk.word_tokenize(s)
    s_tokenized_stopwords_removed = [word for word in s_tokenized if word not in stopwords]
    s_tokenized_stopwords_removed_joined = " ".join(s_tokenized_stopwords_removed)
    return s_tokenized_stopwords_removed_joined


def tokenize_tag_combine(s):
    s_tokenized = nltk.word_tokenize(s)
    s_pos_tagged = nltk.pos_tag(s_tokenized)

    new_s = ''
    for word, pos in s_pos_tagged:
        new_s = new_s + word + "_" + pos + " "

    return new_s


def prepare_data():
    # read in data
    df = pd.read_csv("../data/combined.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)

    # get part of speech
    test_string = "This is a test string."
    test_string = pre_process(test_string)
    test_string = tokenize_tag_combine(test_string)

    df['pos_tagged'] = df['text'].apply(tokenize_tag_combine)
    df.to_csv("../data/combined_pos_tagged.csv")
    print(df.head())


def main():
    n_gram_size = 1
    df = pd.read_csv("../data/combined_pos_tagged.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df[df['label'].notna()]

    print(df.columns)
    print(df.head())

    X = df['pos_tagged'].values.astype('U')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    lr_classifier = Pipeline([('tvect', TfidfVectorizer(ngram_range=(1, n_gram_size))),
                              ('ttrans', TfidfTransformer()),
                              ('scaler', preprocessing.StandardScaler(with_mean=False)),
                              ('cls', LogisticRegression(class_weight='balanced', dual=True, solver='liblinear', max_iter=10000))])

    lr_classifier.fit(X_train, y_train)
    # dump(lr_classifier, '../models/model_{}-gram.pkl'.format(n_gram_size))
    # dump(lr_classifier, '../models/model_{}-gram.joblib'.format(n_gram_size))
    y_pred = lr_classifier.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

    # TODO Need to find a way to oneHotEncode labels
    # print("ROC score: {}".format(roc_auc_score(y_test, y_pred, multi_class='ovr')))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print()

    lr_classifier = load('../models/model_{}-gram.joblib'.format(n_gram_size))

    new_test_string = "exposure to violent video games causes at least a temporary increase in aggression"
    print("Raw string input: {}".format(new_test_string))
    new_test_string = pre_process(new_test_string)
    new_test_string = tokenize_tag_combine(new_test_string)
    new_val = lr_classifier.predict_proba([new_test_string])
    print("Preprocessed string input: {}".format(new_test_string))
    print()
    new_val_df = pd.DataFrame(new_val, columns=['claim', 'premise', 'both', 'neither'])
    print("Probability matrix:")
    print(new_val_df.head())
    print()
    print("Probability sum: {}".format(np.sum(new_val)))
    print()
    print("Cross validating")

    accuracy = cross_val_score(lr_classifier, X, y, scoring='accuracy', cv=10)
    print("Cross validation array: {}".format(accuracy))
    print("Cross validation average: {}".format(np.mean(accuracy) * 100))


if __name__ == '__main__':
    time_start = perf_counter()
    main()
    time_stop = perf_counter()
    print("Time to complete: {} seconds".format(time_stop - time_start))
