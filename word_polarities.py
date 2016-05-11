from data_processing import parse_NRC

def get_neg_words(word_sentiments):
    neg_words = []
    for word in word_sentiments.keys():
        if word_sentiments[word] == "negative":
            neg_words.append(word)
    return neg_words

def get_pos_words(word_sentiments):
    pos_words = []
    for word in word_sentiments.keys():
        if word_sentiments[word] == "positive":
            pos_words.append(word)
    return pos_words

word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
neg_words = get_neg_words(word_sentiments)
pos_words = get_pos_words(word_sentiments)
