from data_processing import parse_stanford, parse_goldstandard, parse_NRC, parse_episodes
from maxent import train_maxent, run_maxent
from bag_of_words import bag_of_words
import random
import pickle
import json


def compare_scores(episode, score1="maxent_score", score2="gold_score"):
    script = episode['script']
    agreement = 0.0
    total = 0.0
    for line in script:
        total += 1
        if score1 not in line or score2 not in line:
            agreement += 0
        elif line[score1] == line[score2]:
            agreement += 1
        elif line[score1] == "neutral":
            agreement += 0.0
    return agreement/total

def test_features():
    training_set = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt", "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    weights = train_maxent(training_set)
    sorted_weights = {sentiment: sorted(weights[sentiment].iteritems(), key=lambda x:x[1], reverse=True) for sentiment in weights}
    for i in range(10000, 130000, 10000):
        new_weights = {"positive": {}, "negative": {}, "neutral": {}}
        for sentiment in sorted_weights:
            new_weights[sentiment] = {w[0]:weights[sentiment][w[0]] for w in sorted_weights[sentiment][:i-1]}

        # testing
        episode = parse_goldstandard("data/s1e9_gold.txt", 1, 9)
        # bag of words
        word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
        bag_of_words(episode, word_sentiments)
        # maxent
        run_maxent(episode, new_weights)

        print "%s max_ent vs gold: %s" % (i, compare_scores(episode, score1="maxent_score", score2="gold_score"))
        print "%s bow vs gold: %s" % (i, compare_scores(episode, "bow_score", score2="gold_score"))


def save_weights():
    training_set = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt", "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    weights = train_maxent(training_set)
    sorted_weights = {sentiment: sorted(weights[sentiment].iteritems(), key=lambda x:x[1], reverse=True) for sentiment in weights}
    new_weights = {"positive": {}, "negative": {}, "neutral": {}}
    for sentiment in sorted_weights:
        new_weights[sentiment] = {w[0]:weights[sentiment][w[0]] for w in sorted_weights[sentiment][:70000]}
    pickle.dump(new_weights, open("weights_optimized.pkl", "wb"))


def test_wwscripts():
    wwscripts = parse_episodes("data/wwscripts.json")
    weights = pickle.load(open("weights_optimized.pkl", "rb"))
    for episode in wwscripts:
        run_maxent(episode, weights)
        pickle.dump(episode, open("data/episode_maxents/s%se%s.pkl" % (episode['season'], episode['episode']), "w"))

def character_sentiment_in_episode(character, episode, score="maxent_score"):
    script = episode['script']
    the_lines = []
    positive = 0.0
    negative = 0.0
    neutral = 0.0
    total = 0.0
    for line in script:
        if character in line['character'] and line[score]:
            total += 1
            if line[score] == "positive":
                positive += 1
            if line[score] == "negative":
                negative += 1
            if line[score] == "neutral":
                neutral += 1
    return (positive, neutral, negative)


if __name__ == "__main__":
    # test_features()
    episode = parse_goldstandard("s1e9.txt", 1, 9)
    weights = pickle.load(open("weights_optimized.pkl", "rb"))
    run_maxent(episode, weights)
    print "TOBY", character_sentiment_in_episode("TOBY", episode), character_sentiment_in_episode("TOBY", episode, "gold_score")
    print "JOSH", character_sentiment_in_episode("JOSH", episode), character_sentiment_in_episode("JOSH", episode, "gold_score")
    print "SAM", character_sentiment_in_episode("SAM", episode), character_sentiment_in_episode("SAM", episode, "gold_score")
    print "MANDY", character_sentiment_in_episode("MANDY", episode), character_sentiment_in_episode("MANDY", episode, "gold_score")
    print "BARTLET", character_sentiment_in_episode("BARTLET", episode), character_sentiment_in_episode("BARTLET", episode, "gold_score")
    print "C.J", character_sentiment_in_episode("C.J.", episode), character_sentiment_in_episode("C.J.", episode, "gold_score")
