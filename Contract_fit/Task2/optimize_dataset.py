# !/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################
# Author: Jaya Ram Kollipara                               #
# Task: Contract.fit - Candidates Filtering Assignment     #
# Start Date: 15/10/2019                                   #
# End Date: 21/10/2019                                     #
# File: optimize_dataset                                   #
# Task: Intelligent Batch Sampling - Phase 1/3             #
############################################################

import os
import re
import operator
from argparse import ArgumentParser
from collections import OrderedDict
import string
import itertools
import pandas as pd
import numpy as np
from sklearn import utils
from copy import deepcopy
import nltk
from nltk import word_tokenize
from nltk.tag import PerceptronTagger
from nltk.corpus import alpino as alp
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class OptimizeDataSet:
    """
    Object class required to construct the optimal mixture of data set. Performs various operations on the data set,
    Constructs regex, Extracts contextual Information
    """
    def __init__(self, params):
        self.params = params
        self.pos_thr = self.params.positive_size
        self.neg_thr = self.params.negative_size
        self.data_frame = self.read_csv()
        self.df_positives = self.positive_samples()
        self.df_negatives = self.negative_samples()
        self.ignore = list(set(stopwords.words('dutch', 'french')))
        self.ss = SnowballStemmer('dutch', 'french')
        self.tagger = self.train_corpus_to_tag()

    def read_csv(self):
        """
        Loads the .csv file into pandas DataFrame
        :return: Pandas Data Frame
        """
        filename = self.params.data

        data_frame = pd.read_csv(filename, sep=None, engine="python")

        return data_frame

    def positive_samples(self):
        """
        Computes Positive Samples by looking at the Target Column
        :return: Positive Samples - Sliced Data Frame
        """
        neg_index = self.data_frame[self.data_frame['stringgold'].isnull()].index
        positives = self.data_frame.drop(labels=neg_index)
        return positives

    def negative_samples(self):
        """
        Computes Negative Samples by looking at the Target Column
        :return: Negative Samples - Sliced Data Frame
        """
        neg_index = self.data_frame[self.data_frame['stringgold'].isnull()].index
        negatives = self.data_frame.ix[neg_index]
        return negatives

    def lattice_pattern(self, element):
        """
        Computes Regular Expression equivalent for a element
        :param element: Sequence of String  <type: 'str'>
        :return: Regular Expression <type: 'str'>
        """
        pattern = ""
        options = OrderedDict({0: "[A-Z]", 1: "[a-z]", 2: "[0-9]", 3: "\s", 4: "\/", 5: "-", 6: '.'})
        for char in element:
            for index, pat in options.items():
                if char.isalpha():
                    pattern += str(char)
                    break
                if re.search(pat, char):
                    pattern += str(index)
                    break
        repetition_groups = list(self.repetitions(pattern))

        result = self.translate_groups(repetition_groups)
        return result

    @staticmethod
    def repetitions(s):
        """
        Counts the number of occurrences,
        For Example: 222422222 as ('2', '3'), ('4', '1'), ('2', '4')
        :param s: <type: 'str'>
        :return: List of Tuples
        """
        r = re.compile(r"(.+?)\1*")
        for match in r.finditer(s):
            yield (match.group(1), int(len(match.group(0)) / len(match.group(1))))

    @staticmethod
    def translate_groups(groups):
        """
        Translates the sequence to a regex
        For example -  ('2', '3'), ('4', '1'), ('2', '4') --> [0-9]{3}[\/]+[0-9]{4}
        :param groups:
        :return: Regular expression  <type: 'str'>
        """
        result = ""
        for gr, count in groups:
            otr = "" if count < 2 else "{" + str(count) + "}"
            if gr.isalpha():
                result += str(gr)
            elif gr == str(2):
                result += str('[0-9]' + otr)
            elif gr == str(3):
                result += '\s*'
            elif gr == str(4):
                result += '[\/]+'
            elif gr == str(5):
                result += '[-]+'
            elif gr == str(6):
                result += '[.]+'
        return result

    @staticmethod
    def searcher(pattern, sequence):
        """
        Searches for a match of the pattern in the sequence
        :param pattern: regex pattern <type: 'str'>
        :param sequence: String <type: 'str'>
        :return: <type: 'bool'>
        """
        match = re.search(pattern, sequence)
        if match is not None:
            return True
        else:
            return False

    @staticmethod
    def similarity_measure(a, b):
        """
        Calculates the Levenshtein distance between a and b.
        :param a: Regex Pattern A
        :param b: Regex Pattern B
        :return: Integer
        """
        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)
        return current[n]

    def construct_regex_patterns(self, labels):
        """
        Contruct regex patterms for every sample in the labels
        :param labels: Gold labels
        :return: Unique reg-ex patterns, <type: pandas.series>
        """
        return list(set(labels.apply(self.lattice_pattern)))

    def group_similar_regex(self, sorted_patterns):
        """
        Group similar patterns as per similarity measure.
        :param sorted_patterns: Pattern Reliance Dictionary  <type: 'dict'>
        :return: Dictionary with closest pattern to keys  <type: 'dict'>
        """
        pattern_dict = {}
        for key in sorted_patterns:
            pattern_dict[key] = []

        for i in range(0, len(sorted_patterns)):
            for j in range(i + 1, len(sorted_patterns)):
                if self.similarity_measure(sorted_patterns[i], sorted_patterns[j]) <= 3:
                    pattern_dict[sorted_patterns[i]].append(sorted_patterns[j])
        return pattern_dict

    def construct_pattern_reliance_on_pos(self, labels):
        """
        Construct Pattern Reliance Dictionary looking at the positive labels
        Sort patterns as per Coverage on positive samples

        :param labels:  Gold labels <type: 'pandas.series'>
        :return: Sorted by Positive Samples Coverage
        """
        patterns = self.construct_regex_patterns(labels)

        pattern_reliance_dict = {}
        for key in patterns:
            self.df_positives['Match'] = self.df_positives['data'].apply(
                lambda x: self.searcher(key, str(x))).astype(np.int64)
            pattern_reliance_dict[key] = self.df_positives['Match'].sum() / self.df_positives.shape[0]

        # sort the dictionary by value
        pattern_reliance_dict_sorted = OrderedDict(sorted(pattern_reliance_dict.items(), key=operator.itemgetter(1),
                                                          reverse=True))

        self.df_positives = self.df_positives.drop("Match", axis=1)
        return pattern_reliance_dict_sorted

    @staticmethod
    def prune_patterns(patterns):
        """
        Prune Pattern Reliance Dictionary
        Input Dictionary contains reg-ex as keys and closest reg-ex as values. Sorted
        :param patterns:  <type: 'dict'>
        :return:
        """
        temp = deepcopy(patterns)
        for key, value in temp.items():
            if key in patterns.keys():
                for v in value:
                    if v in patterns.keys():
                        del patterns[v]
        return patterns

    @staticmethod
    def get_words(labels):
        """
        Tokenize Text Data
        :param labels: Pandas Column
        :return: List of all the words <type: 'list'>
        """
        docs_pos = []
        docs_pos.extend(word_tokenize(words) for words in labels)
        docs_pos = list(itertools.chain(*docs_pos))

        return docs_pos

    @staticmethod
    def get_frequency(words):
        """
        Calcuate Frequency of word in the corpus
        :param words: Tokens from the corpus <type: 'list'>
        :return: Dictionary with Frequency of every token <type: 'dict'>
        """
        return nltk.FreqDist(words)

    def get_popular_words(self, words, thr):
        """
        Capture Most frequent Words from the corpus as per threshold
        :param words: Tokens from the corpus <type: 'list'>
        :param thr: Threshold to be considered as popular <type: 'int'>
        :return: List of popular words
        """
        pos_words = self.get_frequency(words)
        popular_words = []
        for i in pos_words.items():
            if i[1] >= thr:
                popular_words.append(i[0])
        return popular_words

    @staticmethod
    def train_corpus_to_tag():
        """
        Train tagger on Alpino Corpus
        :return: model tagger  <type: 'model'>
        """
        alp_tagged_sent = list(alp.tagged_sents())
        tagger = PerceptronTagger(load=False)
        tagger.train(alp_tagged_sent)
        return tagger

    def pos_tagging(self, words):
        """
        Parts of Speech tagging on Corpus - Dutch and French Corpus Based
        :param words: <type: 'list'>
        :return: tuple (word, parts_of_speech_tag)   <type: 'tuple'>
        """
        tagged_pos_words = self.tagger.tag(words)
        return tagged_pos_words

    def clean_data(self, doc):
        """
        For every sample in the data set, Output only text information
        :param doc: Sample in a pandas data frame <type: 'str'>
        :return: Only Text  <type: 'str'>
        """
        exl_chars = list(set(string.punctuation))
        exl_chars.append("€")
        doc = re.sub(r'[^\x00-\x7f]', r'', doc)
        doc = re.sub("[\w.-]+@[\w.-]+", " ", str(doc))
        doc = re.sub("\d", " ", str(doc))
        doc = ''.join([ch for ch in doc if ch not in exl_chars])
        words = []
        for i in word_tokenize(doc):
            if i not in self.ignore:
                if len(i) >= 2:  # standalone letters do not add any value
                    i = self.ss.stem(i)
                    words.append(i)
        doc = ' '.join(list(set(words)))
        return doc

    @staticmethod
    def construct_vocab(tagged_pos_words):
        """
        Vocabulary based on Positve Words from the Data Set
        :return: List of words <type: 'list'>

        """
        # Context Selection

        filtered_tag_pos_words_nouns = []

        for word in tagged_pos_words:
            if word[1] == 'noun':
                filtered_tag_pos_words_nouns.append(word[0])
        vocab_pos = list(set(filtered_tag_pos_words_nouns))
        return list(set(vocab_pos))


def optimize(params):
    """
    Calculates the optimal Data Set
    Work Flow
    1. Computes the positive and negative samples from the data set
    2. Computes regex patterns on unique positive labels
    3. Prunes the regex patterns based on levenshtein  distance.
    4. Computes score of every sample in the negative and positive data set.
        1. Score - Number of patterns matched by the sample
        2. More the Score, more likely is the negative sample closest to the positive sample
    5. Compute Contextual Information
        1. Construct Vocabulary with most frequent Nouns.
        2. Scores samples, if word in the text contains a word in list vocabulary
    6.  A new Column - Score is added to keep track of number of hits per sample by
    patterns and vocabulary checks.
    7. Order Positive and Negative Samples based on Score value.
    8. Slice the data set according to threshold and return shuffled version of the joint samples

    :param params: .csv file, Required number of positive  and negative samples in the Optimal Mixture <type: 'str'>
    :return: Optimal Mixture of Data Set <type: Pandas DataFrame>
    """

    opt = OptimizeDataSet(params)

    pos_thr = opt.pos_thr
    neg_thr = opt.neg_thr

    df_positives = opt.df_positives
    df_negatives = opt.df_negatives

    if pos_thr > df_positives.shape[0]:
        print("Threshold Higher than the Positive Samples")
        print("Available Positive Samples: ", df_positives.shape[0])
        return

    if neg_thr > df_negatives.shape[0]:
        print("Threshold Higher than the Negative Samples")
        print("Available Negative Samples: ", df_negatives.shape[0])
        return

    # populate pattern reliance Dictionary
    reliance_dict = opt.construct_pattern_reliance_on_pos(df_positives['stringgold'])
    sorted_patterns_per_pos_score = list(reliance_dict.keys())
    reliance_dict = opt.group_similar_regex(sorted_patterns_per_pos_score)
    pruned_patterns = opt.prune_patterns(reliance_dict)

    df_positives["clean_text"] = df_positives["data"].apply(opt.clean_data)
    docs_pos = opt.get_words(df_positives["clean_text"])
    popular_words = opt.get_popular_words(docs_pos, 1000)
    tagged_pos_words = opt.pos_tagging(popular_words)

    vocabulary = opt.construct_vocab(tagged_pos_words)

    def is_exists_vocabulary(doc):
        """
        Chek if the word exists in the vocabulary list
        :param doc: Sample from the data
        :return: <type: 'bool'>
        """
        words = word_tokenize(doc)
        for w in words:
            if w in vocabulary:
                return True
            else:
                return False

    # feature to measure the number of hits by a pattern x in pruned_patterns
    df_positives["Score"] = np.zeros((df_positives.shape[0], 1), dtype=np.int64)
    df_negatives["Score"] = np.zeros((df_negatives.shape[0], 1), dtype=np.int64)

    for key in pruned_patterns:
        df_positives['Match'] = df_positives['data'].apply(lambda x: opt.searcher(key, str(x))).astype(np.int64)
        df_negatives["Match"] = df_negatives['data'].apply(lambda x: opt.searcher(key, str(x))).astype(np.int64)
        if (df_negatives['Match'].sum() / df_negatives.shape[0]) > 0.0:  # no hits
            df_positives["Score"] += df_positives["Match"]
            df_negatives["Score"] += df_negatives["Match"]
        df_negatives = df_negatives.drop('Match', axis=1)
        df_positives = df_positives.drop('Match', axis=1)

    # perform if text contains any words form the vocabulary
    df_negatives['Match'] = df_negatives['data'].apply(is_exists_vocabulary).astype(np.int64)
    df_positives['Match'] = df_positives['data'].apply(is_exists_vocabulary).astype(np.int64)

    df_positives["Score"] += df_positives["Match"]
    df_negatives["Score"] += df_negatives["Match"]
    df_negatives = df_negatives.drop('Match', axis=1)
    df_positives = df_positives.drop('Match', axis=1)

    # sort the data frame based on number of hits - Descending Order

    # Fix
    # eliminate greater than 25% pos values
    thr = int(df_positives.Score.quantile([0.25]).values[0])
    df_negatives['Match'] = df_negatives['Score'].apply(lambda x: 1 if x > thr else 0)

    index_to_drop = df_negatives.index[df_negatives['Match'] == 1].tolist()
    df_negatives = df_negatives.drop(index=index_to_drop, axis=0)
    df_negatives = df_negatives.drop('Match', axis=1)
    df_positives = df_positives.drop('clean_text', axis=1)
    df_positives = df_positives.sort_values(by=['Score'], ascending=False)
    df_negatives = df_negatives.sort_values(by=['Score'], ascending=False)

    # slice the data set based on pos_thr, neg_thr
    df_positives = df_positives.iloc[:pos_thr]
    df_negatives = df_negatives.iloc[:neg_thr]

    df = utils.shuffle(df_positives.append(df_negatives))

    return df


if __name__ == "__main__":
    arp = ArgumentParser()
    arp.add_argument('data', nargs="?", default=os.path.realpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', "data", 'BIO_val.csv')),
                     help='input CSV with segments and gold')
    arp.add_argument("-p", dest="positive_size", default=25000, type=int,
                     help="estimate number of + samples in final csv")
    arp.add_argument("-n", dest="negative_size", default=50000, type=int,
                     help="estimate number of - samples in final csv")
    args = arp.parse_args()

    data = optimize(args)

    if data is not None:
        data.to_csv(os.path.join(os.path.dirname(args.data), os.path.basename(args.data).split(".")[0] + "_opt.csv"),
                    encoding='utf-8', header=True, index=True)
