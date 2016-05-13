"""
maxent.py
Describes all the functions used to model a MaxEnt classifier
"""
from data_processing import parse_stanford
from collections import defaultdict
from features import calculate_features
from math import exp, log
import pickle


def most_probable_class(text, weights):
    """ 
    text is a string
    weights is a dictionary of form
    { 'positive': {key: float},
    'negative': {key: float},
    'neutral': {key: float} }

    given weights corresponding to three classes (positive, negative, neutral),
    this function calculates the feature values for the given text and uses
    them and the three kinds of weights to find which class is the most
    probable for the text.

    probability(class|text) = argmax_c (weights_c dot features)
    """

    pos_weights = weights['positive']
    neg_weights = weights['negative']
    neu_weights = weights['neutral']
    features = calculate_features(text)
    pos_numerator = 0.0
    neg_numerator = 0.0
    neu_numerator = 0.0
    denominator = 0.0
    for f in features:
        if f in pos_weights and f in neg_weights and f in neu_weights:
            pos_numerator += pos_weights[f] * features[f]
            neg_numerator += neg_weights[f] * features[f]
            neu_numerator += neu_weights[f] * features[f]
            denominator += pos_numerator + neg_numerator + neu_numerator
        else:
            pos_numerator += 0
            neg_numerator += 0
            neu_numerator += 0
            denominator += pos_numerator + neg_numerator + neu_numerator

    pos_prob = ("positive", exp(pos_numerator))# /exp(denominator))
    neg_prob = ("negative", exp(neg_numerator))# /exp(denominator))
    neu_prob = ("neutral", exp(neu_numerator))# /exp(denominator))
    return max(neu_prob, neg_prob, pos_prob, key=lambda x: x[1])



def train_maxent(training_set):
    """
    training set is of the form [string: positive|negative|neutral

    trains weights for the maxent classifier by calculating the features for
    every phrase in the training set, and then using a Maximum Likelihood
    Estimate for calculating the optimum weights for each class for the
    classifier
    """
    # calculate features
    feature_counts = {}
    for phrase in training_set:
        features = calculate_features(phrase)
        sentiment = training_set[phrase]
        # f is a feature name
        for f in features:
            if f not in feature_counts:
                feature_counts[f] = defaultdict(float)
                feature_counts[f][sentiment] = features[f]
            else:
                # tally up the total sum of this specific feature for this
                # sentiment
                feature_counts[f][sentiment] += features[f]
    posweights = defaultdict(float)
    negweights = defaultdict(float)
    neuweights = defaultdict(float)
    for f in feature_counts:
        # maximum likelihood estimate 
        denominator = (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
        posweights[f] = feature_counts[f]["positive"] / denominator
        negweights[f] = feature_counts[f]["negative"] / denominator
        neuweights[f] = feature_counts[f]["neutral"] / denominator
            

    return {"positive": posweights,
            "negative": negweights,
            "neutral": neuweights
            }


def run_maxent(episode, weights):
    """ 
    uses maxent weights to classify an episode
    episode is of the format 
    { 'season': int,
      'episode': int,
      'script': [{"character": string, 
                  "text": string, 
                  score: "positive"|"negative"|"neutral"},...]
    """
    script = episode['script']
    for i, line in enumerate(script):
        line = line['text']
        episode['script'][i]['maxent_score'] = most_probable_class(line, weights)[0]
        

if __name__ == "__main__":
    # # import training data
    # training_set = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt",
    #         "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    # # tokenize/preprocess training data. Probs not.
    # # calculate feature dictionary for each text in training data
    # # print d.keys()[0], " ---\t",  d[d.keys()[0]]
    # 
    # # print calculate_features(d.keys()[0])
    # feature_counts = {}
    # for phrase in training_set:
    #     features = calculate_features(phrase)
    #     polarity = training_set[phrase]
    #     for f in features:
    #         if f not in feature_counts:
    #             feature_counts[f] = defaultdict(float)
    #             feature_counts[f][polarity] = features[f]
    #         else:
    #             feature_counts[f][polarity] += features[f]
    # posweights = defaultdict(float)
    # negweights = defaultdict(float)
    # neuweights = defaultdict(float)
    # for f in feature_counts:
    #     posweights[f] = feature_counts[f]["positive"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
    #     negweights[f] = feature_counts[f]["negative"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
    #     neuweights[f] = feature_counts[f]["neutral"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
    # pickle.dump((posweights,negweights,neuweights), open("weights.pkl", "wb"))
    # calculate weights with each feature dicionary as input
    # return weights
    posweights, negweights, neuweights = pickle.load(open("weights.pkl", "rb"))
    print most_probable_class("hello what is going on", posweights, negweights,
            neuweights)
    print most_probable_class("I hate you so much", posweights, negweights,
            neuweights)
    print most_probable_class("I love you so much", posweights, negweights,
            neuweights)
    print most_probable_class("Obaid Farooqui", posweights, negweights, neuweights)
