"""
data_processing.py
This file describes the all the code needed to parse our data files
"""
from collections import defaultdict
from nltk.tokenize import word_tokenize
import json
import re

def parse_NRC(fname):
    """ 
    process the NRC emotion lexicon and convert to a dictionary.
    Returns: {string: "positive"|"negative"|"neutral"}
    """
    f = open(fname)
    word_sentiments = defaultdict(lambda: str("neutral"))
    '''
    { word: "pos"/"neg"/"neu" }
    '''
    for line in f:
        line = line.strip().split("\t")
        word = line[0]
        tag = line[1]
        is_tag = int(line[2])
        if tag == "negative" and is_tag == 1:
            word_sentiments[word] = tag
        if tag == "positive" and is_tag == 1:
            if word_sentiments[word] == "negative":
                word_sentiments[word] = "neutral"
            else:
                word_sentiments[word] = tag

    return word_sentiments


def parse_stanford(dictionary_fname, labels_fname):
    """ Parses the Stanford Sentiment database, given the dictionary and labels
    file"""
    dictionary = {}
    labels = {}

    with open(dictionary_fname, "r") as r:
        for line in r:
            line = line.split("|")
            phrase = line[0].strip()
            phrase_id = line[1].strip()
            dictionary[phrase_id] = phrase

    with open(labels_fname, "r") as r:
        r.readline()
        for line in r:
            line = line.split("|")
            phrase_id = line[0].strip()
            sentiment = float(line[1].strip())
            if sentiment >= 0.6:
                sentiment = "positive"
            elif sentiment <= 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            labels[phrase_id] = sentiment
    new_dict = {}
    for key in dictionary.keys():
        new_dict[dictionary[key]] = labels[key]

    # returns a mapping of phrases to sentiments
    return new_dict


def parse_episodes(wwscripts_json):
    """
    reads a json file of scripts of "The West Wing" and returns them 
    
    Returns: a list of dictionaries of the form 
        {'episode': X, 
        'season': X,
        'script': [{ "character": NAME, "text": LINE_STRING }, ... ]
        }
    """
    wwscripts = json.load(open(wwscripts_json))
    new_wwscripts = []
    for episode in wwscripts:
        ''' episode format:
        {'episode': X, 
        'season': X,
        'script': [{ "character": NAME, "text": LINE_STRING }, ... ]
        }
        '''
        new_script = []
        if len(episode['script']) > 0:
            script = episode['script'][0].split("\n\n")
            for line in script:
                # print line.replace("\n", "\\n")
                line = re.split("([A-Z.^*]+)\n", line, 1)
                if len(line) == 3 and line[0] == '':
                    new_script.append({
                        'character': line[1],
                        'text': line[2].replace("\n", " ")})

            new_episode = {'episode': episode['episode'],
                    'season': episode['season'],
                    'script': new_script}
            new_wwscripts.append(new_episode)

    return new_wwscripts


def parse_goldstandard(filename, season_num, episode_num):
    """
    parses a west wing script that contains gold standard labels
    returns: an episode instance appended with gold_scores
    """
    f = open(filename).read().split("\n\n")
    episode = {"episode": episode_num, 
            "season": season_num,
            "script" : []}
    for line in f:
        line = line.split("\n")
        if len(line) == 3:
            character = line[0].strip() 
            sentiment = line[1].strip()
            if sentiment == "pos":
                sentiment = "positive"
            elif sentiment == "neg":
                sentiment = "negative"
            else:
                sentiment = "neutral"
            text = line[2].strip()
            episode["script"].append({
                "character": character,
                "text": text,
                "gold_score": sentiment})
    return episode



if __name__ == "__main__":
    # s = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    # d = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt",
    #        "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    # print parse_goldstandard("data/s1e9_gold.txt", 1, 9)
