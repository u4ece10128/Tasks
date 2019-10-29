############################################################
# Author: Jaya Ram Kollipara                               #
# Task: Contract.fit - Candidates Filtering Assignment     #
# Start Date: 26/09/2019                                   #
# End Date: 02/10/2019                                     #
# File: Train                                              #
############################################################


import pandas as pd
from sklearn import model_selection, metrics
import numpy as np
# NLP Modules
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import string
from sklearn.utils import shuffle
from nltk.corpus import alpino as alp
from nltk.tag import PerceptronTagger
from collections import Counter
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import tensorflow as tf
import pickle
import sys

# import data
# print("Enter File Name(*.csv)")
# file_input = input()


def main(file_input):
    data_df = pd.read_csv(str(file_input) + '.csv')
    data_df = shuffle(data_df)

    print("Loaded .csv file Successfully")

    print("Total Number of Samples:", data_df.shape[0])
    print("Total Number of Features:", data_df.shape[1])

    # Missing Values
    # column with maximum missing values

    def missing_value(data_df):
        while data_df.isnull().sum().values.sum() != 0:
            col_with_missing_val = (data_df.isnull().sum()).argmax()
            data_df = data_df[data_df[col_with_missing_val].notnull()]  # drop corresponding rows that has NaN values
            print("Missing Values in Features:", col_with_missing_val)
        return data_df

    #  Missing Value Treatment:
    print("Missing Value Treatment : Start")
    data_df = missing_value(data_df)
    print("Missing Value Treatment : Stop")
    print("Total Number of Samples:", data_df.shape[0])
    print("Total Number of Features:", data_df.shape[1])

    # pattern matcher for candidate feature
    #  newly Added Features : Dates format, currency format, number of digits per candidate, number of separators
    # per candidate
    print("Computing Pattern Transformers: Start")

    pattern_strictlyDigits = "^[0-9]*$"
    pattern_endWithCharacters = "^\d*[\/.,@$!)(]$"  # Only digits + end with special characters
    pattern_telephone = "^0[0-9]{12}$"
    pattern_vat = "^0?[0-9]{9}$"
    pattern_date = '^[0-3]?[0-9](\/|\,|\.|\-){1}[0-9]?[0-9](\/|\,|\.|\-){1}[0-2][0-9]{1,3}$'

    pattern_currency_1 = '^[0-9]\.[0-9]+\,[0-9]*$'  # captures ddddd,dddd
    pattern_currency_2 = '^[0-9]+\,[0-9]+$'
    data_df['currency_filter'] = data_df['candidate'].str.contains(pattern_currency_1, regex=True).astype(np.int64)\
                                 | data_df['candidate'].str.contains(pattern_currency_2, regex=True).astype(np.int64)

    data_df['dates_filter'] = data_df['candidate'].str.contains(pattern_date, regex=True).astype(np.int64)
    data_df["Is_strictly_Digits"] = data_df["candidate"].str.contains(pattern_strictlyDigits, regex=True).astype(np.int64)
    data_df["endWithCharacters"] = data_df["candidate"].str.contains(pattern_endWithCharacters, regex=True).astype(np.int64)
    data_df["Number_of_Digits"] = data_df['candidate'].apply(lambda x: len(re.sub("\W", "", x)))
    data_df["Number_of_Separators"] = data_df['candidate'].apply(lambda x: len(re.sub("\w", "", x)))
    data_df["Length_of_Candidate"] = data_df['candidate'].apply(lambda x: len(x))

    # included the country code
    data_df["Telephone"] = data_df["candidate"].str.contains(pattern_telephone, regex=True).astype(np.int64)
    # VAT number contains 9 to 10 digits
    data_df["VATNumber"] = data_df["candidate"].str.contains(pattern_vat, regex=True).astype(np.int64)

    # drop blacklisted variables
    dates_index = data_df.index[data_df['dates_filter'] == 1].tolist()
    data_df = data_df.drop(index=dates_index, axis=0)
    data_df = data_df.drop("dates_filter", axis=1)
    currency_index = data_df.index[data_df['currency_filter'] == 1].tolist()
    data_df = data_df.drop(index=currency_index, axis=0)
    data_df = data_df.drop(["currency_filter"], axis=1)
    telephone_index = data_df.index[data_df['Telephone'] == 1].tolist()
    data_df = data_df.drop(index=telephone_index, axis=0)
    data_df = data_df.drop(["Telephone"], axis=1)
    vat_index = data_df.index[data_df['VATNumber'] == 1].tolist()
    data_df = data_df.drop(index=vat_index, axis=0)
    data_df = data_df.drop(["VATNumber"], axis=1)
    vat_index = data_df.index[data_df['endWithCharacters'] == 1].tolist()
    data_df = data_df.drop(index=vat_index, axis=0)
    data_df = data_df.drop(["endWithCharacters"], axis=1)

    print("Computing Pattern Transformers: Stop")

    # NLP Techniques:
    # Tokenization, Stemming, lemmatization, Frequency Distribution, Bag of words approach

    # Combine three text columns to single column - This columns contains he full text
    data_df["Text"] = data_df["line_before"] + data_df["line_at"] + data_df["line_after"]

    print("Computing Context Transformers: Start")

    # Context Transformers
    def email_match(doc):
        match = re.search(r'[\w\.-]+@[\w\.-]+', str(doc))
        if match != None:
            return 1
        else:
            return 0

    data_df["Number_of_Characters_Text"] = data_df["Text"].apply(lambda x: len(re.sub("[^a-z]", "", str(x))))
    data_df["Number_of_Digits_Text"] = data_df["Text"].apply(lambda x: len(re.sub("[^0-9]+", "", str(x))))
    data_df["Number_of_Separators_Text"] = data_df["Text"].apply(
        lambda x: len((re.sub("[\w]+", "", str(x))).replace(" ", "")))
    data_df["Is_Email_Exists"] = data_df["Text"].apply(email_match)  # place 1 everywhere email found else 0
    data_df["Number_of_spaces"] = data_df["Text"].apply(lambda x: str(x).count(' '))  # counts number of spaces,

    # Clean Data - Tokenization, Stop word check, Size filter, Stemming - Dutch Language
    ss = SnowballStemmer("dutch", "french")

    def clean_data(doc):
        ignore = list(set(stopwords.words('dutch', 'french')))  # ignore the list of stopwords
        exl_chars = list(set(string.punctuation))
        exl_chars.append('€')
        # remove email ids to avoid conflicts in vocabulary construction
        doc = re.sub("[\w\.-]+@[\w\.-]+", " ", str(doc))
        doc = re.sub("\d", " ",  str(doc))
        doc = ''.join([ch for ch in doc if ch not in exl_chars])
        words = []
        for i in word_tokenize(doc):  # tokenization
            if i not in ignore:
                if len(i) >=2: # standalone letters do not add any value
                    i = ss.stem(i)
                    words.append(i)
        doc = ' '.join(list(set(words)))
        return doc

    print("Cleaning Text Data: Start")
    data_df["Text"] = data_df["Text"].apply(clean_data)  # tokenize, stem and lammetize
    print("Cleaning Text Data: Stop")

    print("Computing POS Vectors: Start")

    # training_corpus = alp.tagged_sents()
    alp_tagged_sent = list(alp.tagged_sents())
    tagger = PerceptronTagger(load=False)
    tagger.train(alp_tagged_sent)

    def count_adj(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >=3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >=3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word,tag in tags)
        count_adj_adv = counts['adv'] + counts['adj']
        return count_adj_adv

    def count_nn(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >=3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >=3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word,tag in tags)
        count_nn = counts['noun']
        return count_nn

    def count_verb(doc):
        tags = tagger.tag(doc.split())
        for tup in tags:
            first_3_characters = tup[0][:3]
            last_3_characters = tup[0][3:]
            if len(tags[0]) >=3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
                tags.remove(tup)
            if len(tags[0]) >=3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
                tags.remove(tup)
        counts = Counter(tag for word,tag in tags)
        count_verb = counts['verb']
        return count_verb

    data_df["Adv_Adj_Count"] = data_df["Text"].apply(count_adj)
    data_df["NN_count"] = data_df["Text"].apply(count_nn)
    data_df["Verb_count"] = data_df["Text"].apply(count_verb)

    print("Computing POS Vectors: Stop")

    print("Computing Vocabulary: Start")

    # store all the words in positive class and negative in two separate lists
    docs_pos = []

    docs_pos.extend(word_tokenize(words) for words in data_df.Text[data_df.gold == 1])

    docs_pos = list(itertools.chain(*docs_pos))

    # Clean text data - remove words like --- iiiiiii, hhhhhccchhhh, abvwwwwwcgdccc
    for i in docs_pos:
        first_3_characters = i[:3]
        last_3_characters = i[-3:]
        if len(i) >= 3 and first_3_characters[0] == first_3_characters[1] == first_3_characters[2]:
            docs_pos.remove(i)
        if i in docs_pos and len(i) >= 3 and last_3_characters[0] == last_3_characters[1] == last_3_characters[2]:
            docs_pos.remove(i)

    print("Positve class words are stored successfully")

    all_words_pos = nltk.FreqDist(docs_pos)

    print("Computing vocabulary based on Positive Class")
    # find popular words, popular equals more than 25 times in the corpus
    popular_pos_words = []
    for i in all_words_pos.items():
        if i[1] >= 25:
            popular_pos_words.append(i[0])

    # Filter nouns from the popular positive class words
    tagged_pos_words = tagger.tag(popular_pos_words)
    filtered_tag_pos_words_nouns = []
    for word in tagged_pos_words:
        if word[1] == 'noun':
            filtered_tag_pos_words_nouns.append(word[0])
    vocab_pos = list(set(filtered_tag_pos_words_nouns))
    vocabulary = list(set(vocab_pos))

    # save vocabulary
    with open("vocab.txt", "wb") as fp:
        pickle.dump(vocabulary, fp)

    print("Computing Vocabulary: Stop")

    print("Length of Vocabulary: ", len(vocabulary))

    print("Computing Bag of Words Vectors: Start")

    def build_features(doc):
        vector = np.zeros((1, len(vocabulary)), dtype=np.int64)
        for w in word_tokenize(doc):
            for idx, vocab in enumerate(vocabulary):
                if vocab == w:
                    vector[0][idx] += 1
        return vector

    bag_vectors = data_df["Text"].apply(build_features)

    feature_vectors = np.zeros((data_df.shape[0], len(vocabulary)), dtype=np.int64)
    for pos, index in enumerate(data_df.index.values):
        feature_vectors[pos, :] = bag_vectors[index]

    cols = ["BOW_" + str(col) for col in range(0, len(vocabulary))]
    for col_index, col in enumerate(cols):
        data_df[col] = feature_vectors[:, col_index].reshape(data_df.shape[0], 1)

    print("Computing Bag of Words Vectors: Stop")

    print("Computing Context Transformers: Stop")

    print("Computing Location Transformers: Start")

    data_df["location_page_nr"] = data_df["page_nr"].apply(lambda x: 100 if x >= 50 else x)
    data_df["location_line_nr"] = data_df["line_nr"].apply(lambda x: 100 if x >= 50 else x)

    print("Computing Location Transformers: Stop")

    print("Total Number of Newly Added Features:", data_df.shape[1] - 7)

    print("Building ML - Neural Network Model: Start")

    X = data_df.drop(["candidate", "Text", "gold","label", "line_after", "line_at", "line_before",
                      "line_nr", "page_nr"], axis=1)
    y = data_df.gold
    #  Normalisation
    X = (X-X.mean(axis=0))/X.std(axis=0)

    def build_model(input_shape):
        model = Sequential()
        model.add(Dense(1024, input_shape=(input_shape,)))
        model.add(Activation('sigmoid'))

        model.add(Dense(512))
        model.add(Activation('sigmoid'))

        model.add(Dense(128))
        model.add(Activation('sigmoid'))

        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])
        return model

    #  Stratified k-Fold
    k_fold_outer = model_selection.StratifiedKFold(n_splits=5)
    scores = []
    split = 0
    for train_index, test_index in k_fold_outer.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        model = build_model(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=5, batch_size=1024, verbose=1)
        results = model.evaluate(X_val, y_val)
        scores.append(results[1])
        split += 1
        del model, history, results

    model = build_model(X.shape[1])
    model.fit(X, y, verbose=0)

    print('Saving the Model *.h5...')
    model.save('model_candidate_filter.h5')

    yHat_proba = model.predict(X)
    yHat = np.copy(yHat_proba)
    yHat[yHat <= 0.5] = 0
    yHat[yHat > 0.5] = 1

    br_score = np.around(metrics.brier_score_loss(y, yHat_proba, pos_label=1), decimals=5)
    print("Storing Results in .csv file")

    confidence = np.zeros((yHat_proba.shape[0], yHat_proba.shape[1]))
    for i in range(0, yHat_proba.shape[0]):
        if yHat_proba[i] <= 0.5:
            confidence[i] = 1 - yHat_proba[i]
        else:
            confidence[i] = yHat_proba[i]

    results_data_frame = pd.DataFrame(columns=["Predictions", "Confidence Level"], index=data_df.index)
    results_data_frame["Predictions"] = yHat.astype(np.int64).ravel()
    results_data_frame["Confidence Level"] = np.around(confidence, decimals=4)
    results_data_frame.to_csv("Results_predictions_confidence_train.csv",  encoding='utf-8', header=True, index=True)

    return np.mean(scores), br_score

if __name__ == "__main__":
    acc, br = main(sys.argv[1])
    print("Accuracy of the Model: CV Accuracy Score", np.around(acc, decimals=5))
    print("Awareness of the Model: Brier Score:", br)
