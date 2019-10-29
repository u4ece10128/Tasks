############################################################
# Author: Jaya Ram Kollipara                               #
# Task: Contract.fit - Candidates Filtering Assignment     #
# Start Date: 26/09/2019                                   #
# End Date: 02/10/2019                                     #
# File: Run                                                #
############################################################


import pandas as pd
import numpy as np
# NLP Modules
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import string
from nltk.corpus import alpino as alp
from nltk.tag import PerceptronTagger
from collections import Counter
import pickle
import tensorflow as tf
import sys


def main(file_input):
    test_data = pd.read_csv(str(file_input) + '.csv')
    # test_data = pd.read_csv(str(file_input) + '.csv', index_col='Unnamed: 0')

    print("Loaded .csv file Successfully")

    print("Missing Value Treatment : Start")
    # missing values Treatment
    while test_data.isnull().sum().values.sum() != 0:
        col_with_missing_val = (test_data.isnull().sum()).argmax()
        test_data = test_data[test_data[col_with_missing_val].notnull()]  # drop corresponding rows that has NaN values
        print(col_with_missing_val)

    print("Missing Value Treatment : Stop")
    print("Total Number of Samples:", test_data.shape[0])
    print("Total Number of Features:", test_data.shape[1])

    print("Computing Pattern Transformers: Start")
    # pattern transformers
    pattern_strictlyDigits = "^[0-9]*$"
    test_data["strictly_Digits"] = test_data["candidate"].str.contains(pattern_strictlyDigits, regex=True).astype(np.int64)
    test_data["Number_of_Digits"] = test_data['candidate'].apply(lambda x: len(re.sub("\W", "", x)))
    test_data["Number_of_Seprators"] = test_data['candidate'].apply(lambda x: len(re.sub("\w", "", x)))
    test_data["Length_of_Candidate"] = test_data['candidate'].apply(lambda x: len(x))

    print("Computing Pattern Transformers: Stop")
    print("Computing Context Transformers: Start")
    # context transformers
    test_data["Text"] = test_data["line_before"] + test_data["line_at"] + test_data["line_after"]

    def email_match(doc):
        match = re.search(r'[\w\.-]+@[\w\.-]+', str(doc))
        if match !=  None:
            return 1
        else:
            return 0

    test_data["Number_of_Characters_Text"] = test_data["Text"].apply(lambda x: len(re.sub("[^a-z]", "", str(x))))
    test_data["Number_of_Digits_Text"] = test_data["Text"].apply(lambda x: len(re.sub("[^0-9]+", "", str(x))))
    test_data["Number_of_Separators_Text"] = test_data["Text"].apply(
            lambda x: len((re.sub("[\w]+", "", str(x))).replace(" ", "")))
    test_data["Email_Exists"] = test_data["Text"].apply(email_match)  # place 1 everywhere email found else 0
    test_data["Number_of_spaces"] = test_data["Text"].apply(lambda x: str(x).count(' '))  # counts number of spaces

    # Clean Data - Tokenization, Stop word check, Size filter, Stemming - Dutch Language
    ss = SnowballStemmer("dutch", "french")

    def clean_data(doc):
        ignore = list(set(stopwords.words('dutch', 'french')))  # ignore the list of stopwords
        exl_chars = list(set(string.punctuation))
        exl_chars.append('€')
        doc = re.sub("[\w\.-]+@[\w\.-]+", " ", str(doc)) # remove email ids to avoid confiltcs in vaocabulary construction
        doc = re.sub("\d", " ",  str(doc))
        doc = ''.join([ch for ch in doc if ch not in exl_chars])
        words = []
        for i in word_tokenize(doc):  # tokenization
            if i not in ignore:
                if len(i) >= 2:  # standalone letters do not add any value
                    i = ss.stem(i)
                    words.append(i)
        doc = ' '.join(list(set(words)))
        return doc

    test_data["Text"] = test_data["Text"].apply(clean_data)   # tokenize, stem and lammetize

    # training_corpus = alp.tagged_sents()
    alp_tagged_sent = list(alp.tagged_sents())
    tagger = PerceptronTagger(load=False)
    tagger.train(alp_tagged_sent)

    def count_adj(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >= 3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >= 3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word, tag in tags)
        count_adj_adv = counts['adv'] + counts['adj']
        return count_adj_adv

    def count_nn(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >= 3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >= 3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word, tag in tags)
        count_nn = counts['noun']
        return count_nn

    def count_verb(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >= 3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >= 3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word, tag in tags)
        count_verb = counts['verb']
        return count_verb

    test_data["Adv_Adj_Count"] = test_data["Text"].apply(count_adj)
    test_data["NN_count"] = test_data["Text"].apply(count_nn)
    test_data["Verb_count"] = test_data["Text"].apply(count_verb)

    print("Computing Context Transformers: Stop")
    # load the vocabulary
    with open("vocab.txt", "rb") as fp:
        vocabulary = pickle.load(fp)

    print("Computing Bag of Words Vectors: Start")

    def build_features(doc):
        vector = np.zeros((1, len(vocabulary)), dtype=np.int64)
        for w in word_tokenize(doc):
            for i, word in enumerate(vocabulary):
                if word == w:
                    vector[0][i] += 1
        return vector
    bag_vectors = test_data["Text"].apply(build_features)
    feature_vectors = np.zeros((test_data.shape[0], len(vocabulary)), dtype=np.int64)
    for pos, index in enumerate(test_data.index.values):
        feature_vectors[pos,:] = bag_vectors[index]
    cols = ["BOW_" + str(col) for col in range(0, len(vocabulary))]
    for col_index, col in enumerate(cols):
        test_data[col] = feature_vectors[:, col_index].reshape(test_data.shape[0], 1)

    print("Computing Bag of Words Vectors: Stop")

    print("Computing Location Transformers: Start")

    test_data["location_page_nr"] = test_data["page_nr"].apply(lambda x: 100 if x >= 50 else x)
    test_data["location_line_nr"] = test_data["line_nr"].apply(lambda x: 100 if x >= 50 else x)

    print("Computing Location Transformers: Stop")

    print("Loading Model...")
    model = tf.keras.models.load_model('model_candidate_filter.h5')
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])
    print("Loaded Model Successfully!")

    X_test = test_data.drop(["candidate", "Text", "label", "line_after", "line_at", "line_before",
                             "page_nr", "line_nr"], axis=1)

    X_test = (X_test-X_test.mean(axis=0))/X_test.std(axis=0)
    yHat_proba = model.predict(X_test)
    yHat = np.copy(yHat_proba)
    yHat[yHat <= 0.5] = 0
    yHat[yHat > 0.5] = 1

    print("Storing Results in .csv file")

    confidence = np.zeros((yHat_proba.shape[0], yHat_proba.shape[1]))
    for i in range(0, yHat_proba.shape[0]):
        if yHat_proba[i] <= 0.5:
            confidence[i] = 1 - yHat_proba[i]
        else:
            confidence[i] = yHat_proba[i]

    results_data_frame = pd.DataFrame(columns=["Predictions", "Confidence Level"], index=test_data.index)
    results_data_frame["Predictions"] = yHat.astype(np.int64).ravel()
    results_data_frame["Confidence Level"] = np.around(confidence, decimals=4)
    results_data_frame.to_csv("Results_predictions_confidence_run.csv",  encoding='utf-8', header=True, index=True)


if __name__ == "__main__":
    main(sys.argv[1])
