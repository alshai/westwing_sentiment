# defines functions that extract feature values from strings
# ASSUME "text" IS TOKENIZED
from word_polarities import pos_words, neg_words
from nltk.tokenize import word_tokenize
emphatic_words = ["really", "very", "totally", "much"]

def unigrams(text, features):
    for word in text:
        if word in features:
            features[word] += 1
        else:
            features[word] = 1

def bigrams(text, features):
    bgrams = zip(text, text[1:])
    for bgram in bgrams:
        if bgram in features:
            features[bgram] += 1
        else:
            features[bgram] = 1


def num_poswords(text, features):
    num_pos = 0
    for word in text:
        if word in pos_words:
            num_pos  += 1
    features["pos_words"] = num_pos


def num_negwords(text, features):
    num_neg = 0
    for word in text:
        if word in neg_words:
            num_neg  += 1
    features["neg_words"] = num_neg


def num_emphatic_words(text, features):
    num_emph_words = 0
    for word in text:
        if word in emphatic_words:
            num_emph_words +=1 
    features["emphatic_words"] = num_emph_words

def num_exclamation_points(text, features):
    num_excl_points = 0
    for word in text:
        if "!" in word:
            num_excl_points += 1
    features["exclamation_points"] = num_excl_points

feat_funcs = [unigrams, bigrams, num_poswords, num_negwords,
        num_emphatic_words, num_exclamation_points]


def calculate_features(text):
    features = {}
    tokenized_text = word_tokenize(text)
    for func in feat_funcs:
        func(tokenized_text, features)
    return features
