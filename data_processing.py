# all the methods used to parse our data files
from collections import defaultdict

def parse_NRC(fname):
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


if __name__ == "__main__":
    # s = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    d = parse_stanford("data/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt",
            "data/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt")
    print d
