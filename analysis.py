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
    """ episode is a dictionary of format 
    {'season': int, 
    'episode': int,
    'script': [{'maxent_score': 'positive'|'negative'|'neutral',
               'gold_score': 'positive'|'negative'|'neutral',
               'bow_score': 'positive'|'negative'|'neutral'}, ...] 
    }
    score1, score2 specify which score types are being compared

    returns: the percentage of times score1 and score2 agree over all the lines
    """

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

def evaluate_features():
    """
    calculates percent accuracy of our maxent algorithm with varying lengths of
    weights lists (sorted in descending order)
    """
    # training set is from Stanford Sentiment Training Set
    training_set = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt", "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    # train weights for maxent model
    weights = train_maxent(training_set)
    # sort weights in descending order
    sorted_weights = {sentiment: sorted(weights[sentiment].iteritems(), key=lambda x:x[1], reverse=True) for sentiment in weights}

    # evaluate model for the  top i weights, in this range (There should be # ~130000 weights total)
    for i in range(10000, 130000, 10000):
        # get the top i weights
        new_weights = {"positive": {}, "negative": {}, "neutral": {}}
        for sentiment in sorted_weights:
            new_weights[sentiment] = {w[0]:weights[sentiment][w[0]] for w in sorted_weights[sentiment][:i-1]}

        # load the episode that has gold standard features already assigned
        episode = parse_goldstandard("data/s1e9_gold.txt", 1, 9)
        # calculate bag of words sentiments
        word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
        bag_of_words(episode, word_sentiments)
        # calculate maxent sentiments
        run_maxent(episode, new_weights)

        # evaulate maxent and bag_of_words sentiments against baseline
        print "%s max_ent vs gold: %s" % (i, compare_scores(episode, score1="maxent_score", score2="gold_score"))
        print "%s bow vs gold: %s" % (i, compare_scores(episode, "bow_score", score2="gold_score"))


def save_weights():
    """ calculate the weights for the maxent model using the Staford senitment
    training set and save the weights to a pickle file"""
    training_set = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt", "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    weights = train_maxent(training_set)
    sorted_weights = {sentiment: sorted(weights[sentiment].iteritems(), key=lambda x:x[1], reverse=True) for sentiment in weights}
    new_weights = {"positive": {}, "negative": {}, "neutral": {}}
    # get only the top 70000 weights (as determined in evaluate_features())
    for sentiment in sorted_weights:
        new_weights[sentiment] = {w[0]:weights[sentiment][w[0]] for w in sorted_weights[sentiment][:80000]}
    pickle.dump(new_weights, open("weights_optimized.pkl", "wb"))


def test_wwscripts():
    """ assigns labels to all episodes of The West Wing """
    wwscripts = parse_episodes("data/wwscripts.json")
    weights = pickle.load(open("weights_optimized.pkl", "rb"))
    # word_sentiments is a map of words to "positive" or "negative" and is used
    # for the bag_of_words classifier
    word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    for episode in wwscripts:
        run_maxent(episode, weights)
        bag_of_words(episode, word_sentiments)
        pickle.dump(episode, open("data/episode_maxents/s%se%s.pkl" % (episode['season'], episode['episode']), "w"))


def character_sentiment_in_episode(character, episode_script, score="maxent_score"):
    """ 
    character is a string
    episode_script is a list of the format 
    [{"character": string, "text": string, score:"positive"|"negative"|"neutral"},...]
    
    returns the number of positive, neutral and negative lines said by a
    character in a given episode 
    
    """
    positive = 0.0
    negative = 0.0
    neutral = 0.0
    for line in episode_script:
        if character in line['character'] and line[score]:
            if line[score] == "positive":
                positive += 1
            if line[score] == "negative":
                negative += 1
            if line[score] == "neutral":
                neutral += 1
    return (positive, neutral, negative)


def get_episode_sentiment(episode_script, score='maxent_score'):
    """ 
    episode_script is a list of the format 
    [{"character": string, "text": string, score:"positive"|"negative"|"neutral"},...]

    returns the number of positive, neutral and negative lines within a
    given episode
    """
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
    """
    season is of the format
    { episode: [{"character": string, "text": string, score:"positive"|"negative"|"neutral"},...]}

    returns the number of positive, neutral and negative lines within a given season
    """
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


def character_figures(character, series):
    character_sentiments = {}
    for season in series:
        character_sentiments[season] = {}
        for episode in series[season]:
            sentiments = character_sentiment_in_episode(character, series[season][episode])
            if sentiments[0] != 0:
                character_sentiments[season][episode] = sentiments[0] / float(sentiments[0]+sentiments[2])
            else:
                character_sentiments[season][episode] = 0
    for season in character_sentiments:
        print character_sentiments[season]
        sorted_season = sorted(character_sentiments[season].iteritems(), key=lambda x: x[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(zip(*sorted_season)[0], zip(*sorted_season)[1])
        ax.plot(zip(*sorted_season)[0], zip(*sorted_season)[1])
        ax.set_xlim(0, max(zip(*sorted_season)[0]))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode #")
        ax.set_ylabel("Pos/Neg Sentiment Ratio")
        ax.set_title("Pos/Neg Sentiment Ratio for %s for Season %s of 'The West Wing'" % (character, season))

        plt.tight_layout()
        plt.savefig("figures/%s_season_%s.png" % (character, season))

def all_character_figures():
    series = load_episodes()
    for character in ["BARTLET", "LEO"]:
        character_figures(character, series)

    
if __name__ == "__main__":
    # test_wwscripts()
    # weights = pickle.load(open("weights_optimized.pkl", "rb"))
    # single_season_sentiment_figure()
    # all_seasons_sentiment_figure()
    # evaluate_features()
    # save_weights()
    # test_wwscripts()
    series = load_episodes()
    character_figures("BARTLET", series)
