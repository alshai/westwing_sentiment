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

if __name__ == "__main__":
    s = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
