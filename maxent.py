from data_processing import parse_stanford
from collections import defaultdict
from features import calculate_features
from math import exp, log
import pickle


def most_probable_class(text, weights):
    pos_weights = weights[0]
    neg_weights = weights[1]
    neu_weights = weights[2]
    features = calculate_features(text)
    pos_numerator = 0.0
    neg_numerator = 0.0
    neu_numerator = 0.0
    denominator = 0.0
    for f in features:
        pos_numerator += pos_weights[f] * features[f]
        neg_numerator += neg_weights[f] * features[f]
        neu_numerator += neu_weights[f] * features[f]
        denominator += pos_numerator + neg_numerator + neu_numerator
    pos_prob = ("positive", exp(pos_numerator))# /exp(denominator))
    neg_prob = ("negative", exp(neg_numerator))# /exp(denominator))
    neu_prob = ("neutral", exp(neu_numerator))# /exp(denominator))
    return max(neu_prob, neg_prob, pos_prob, key=lambda x: x[1])



def train_maxent(training_set):
    # trains maxent weights give a dictionary of sentences mapped to labels
    feature_counts = {}
    for phrase in training_set:
        features = calculate_features(phrase)
        polarity = training_set[phrase]
        for f in features:
            if f not in feature_counts:
                feature_counts[f] = defaultdict(float)
                feature_counts[f][polarity] = features[f]
            else:
                feature_counts[f][polarity] += features[f]
    posweights = defaultdict(float)
    negweights = defaultdict(float)
    neuweights = defaultdict(float)
    for f in feature_counts:
        posweights[f] = feature_counts[f]["positive"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
        negweights[f] = feature_counts[f]["negative"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])
        neuweights[f] = feature_counts[f]["neutral"] / (feature_counts[f]["positive"]+ feature_counts[f]["negative"] + feature_counts[f]["neutral"])

    return {"positive": posweights,
            "negative": negweights,
            "neutral": neuweights
            }


def run_maxent(episode, weights):
    # updates the episode with maxent scores
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
