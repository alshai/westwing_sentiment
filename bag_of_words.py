""" 
bag_of_words.py
This file contains all functions related to building a "bag of words" model
for sentiment analysis
"""
from data_processing import parse_NRC
from nltk.tokenize import word_tokenize
import pickle


unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"


def convert_to_sentiment(score):
    """
    converts values in the range [0,1] to sentiments
    """
    if score > 0.5:
        return "positive"
    elif score < 0.5:
        return "negative"
    else:
        return "neutral"


def bag_of_words(episode, word_sentiments):
    """
    bag_of_words classifier for sentiment analysis
    episode is a dictionary of format 
    {'season': int, 
    'episode': int,
    'script': [{'text': string}, ...  ]
    }
    word_sentiments is a dictiony of format
    {string: 'positive'|'negative'}


    The bag of words classifier simply averages the amount of positive,
    negative and neutral words found in a string of text and returns the
    sentiment associated with the average. 
    sentiments of value > 0.5 are positive, < 0.5 are negative and == 0.5 are
    neutral
    """
    script = episode["script"]
    for i, line in enumerate(script):
        score = 0.0
        text = word_tokenize(line['text'])
        for word in text:
            if word not in word_sentiments:
                score += 0.5
            elif word_sentiments[word] == "negative":
                score += 0
            elif word_sentiments[word] == "positive":
                score += 1
            elif word_sentiments[word] == "neutral":
                score += 0.5
        score = score / len(text)
        episode['script'][i]['bow_score'] = convert_to_sentiment(score)


if __name__ == "__main__":
    episodes = pickle.load(open("test_scripts.pkl", "rb"))
    word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    bag_of_words(episodes[0], word_sentiments)

