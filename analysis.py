from data_processing import parse_stanford, parse_goldstandard, parse_NRC, parse_episodes
from maxent import train_maxent, run_maxent
from bag_of_words import bag_of_words
import random
import numpy as np
import pickle
import json
from glob import glob
import matplotlib.pyplot as plt


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


def train_weights():
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
    word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    for episode in wwscripts:
        run_maxent(episode, weights)
        bag_of_words(episode, word_sentiments)
        pickle.dump(episode, open("data/episode_maxents/s%se%s.pkl" % (episode['season'], episode['episode']), "w"))


def character_sentiment_in_episode(character, episode_script, score="maxent_score"):
    positive = 0.0
    negative = 0.0
    neutral = 0.0
    total = 0.0
    for line in episode_script:
        if character in line['character'] and line[score]:
            total += 1
            if line[score] == "positive":
                positive += 1
            if line[score] == "negative":
                negative += 1
            if line[score] == "neutral":
                neutral += 1
    return (positive, neutral, negative)


def get_episode_sentiment(episode_script, score='maxent_score'):
    pos = 0
    neg = 0
    neu = 0
    for line in episode_script:
        if line[score] == 'positive':
            pos += 1
        if line[score] == 'negative':
            neg += 1
        if line[score] == 'neutral':
            neu += 1
    return pos, neu, neg


def get_season_sentiment(season, score='maxent_score'):
    pos_neg_ratios = []
    pos = 0
    neg = 0
    for episode in season:
        sentiments = get_episode_sentiment(season[episode])
        pos += sentiments[0]
        neg += sentiments[2]
    return pos/float(pos + neg)

def load_episodes():
    """
    load the episodes in a dictionary of the format:
    { season_num: { 
        episode_num: script: [ 
            {'text': text_of_the_line (string), 
            'character': character_who_says_the_line (string), 
            'maxent_score': sentiment_given_by_maxent, 
            'bow_score': sentiment_given_by_bagofwords, 
            'gold_score': ground_truth_sentiment}, ...  ] } 
    }
    """
    episodes = {}
    for fname in glob("data/episode_maxents/*.pkl"):
        episode = pickle.load(open(fname, "rb"))
        if episode['season'] not in episodes:
            episodes[episode['season']] = {}
        episodes[episode['season']][episode['episode']] = episode['script']
    return episodes


def single_season_sentiment_figure():
    episodes = load_episodes()
    # plot pos/neg ratio over the course of season 1
    pos_neg_ratios = []
    for episode in sorted(episodes[1].keys()):
        sentiments = get_episode_sentiment(episodes[1][episode])
        pos_neg_ratios.append(100 * sentiments[0]/ float(sentiments[0] + sentiments[2]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(1, len(episodes[1]) + 1), pos_neg_ratios)
    ax.plot(range(1, len(episodes[1]) + 1), pos_neg_ratios)
    ax.set_xticks(range(1, len(episodes[1])+1))
    ax.set_xlim(0, len(episodes[1]) + 1)
    ax.set_ylim(0,100)
    ax.set_xlabel("Episode #")
    ax.set_ylabel("Pos/Neg Sentiment Ratio")
    ax.set_title("Pos/Neg Sentiment Ratios for Season 1 Episodes of 'The West Wing'")
    plt.tight_layout()
    plt.savefig("figures/season_1_posneg_ratios.png")

def all_seasons_sentiment_figure():
    episodes = load_episodes()
    # plot pos/neg ratio over the course of all seasons
    season_sentiments = []
    for i in range(5):
        season_sentiments.append(100* get_season_sentiment(episodes[i+1]))

    width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(0,5,1), season_sentiments, width)
    ax.set_ylim(60, 90)
    ax.set_xticks(np.arange(0,5,1) + width /2.)
    ax.set_xticklabels(np.arange(1,6,1))
    ax.set_xlabel("Season #")
    ax.set_ylabel("Pos/Neg Sentiment Ratio")
    ax.set_title("Pos/Neg Sentiment Ratios for First Five Seasons of 'The West Wing'")
    plt.tight_layout()
    plt.savefig("figures/all_seasons_posneg_ratios.png")



if __name__ == "__main__":
    # train_weights()
    # test_wwscripts()
    # weights = pickle.load(open("weights_optimized.pkl", "rb"))
    # single_season_sentiment_figure()
    all_seasons_sentiment_figure()
