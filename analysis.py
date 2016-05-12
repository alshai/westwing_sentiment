from data_processing import parse_goldstandard, parse_NRC
from maxent import run_maxent
from bag_of_words import bag_of_words
import pickle


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
            agreement += 0.75
    return agreement/total



if __name__ == "__main__":
    episode = parse_goldstandard("data/s1e9_gold.txt", 1, 9)
    weights = pickle.load(open("weights.pkl", "rb"))
    run_maxent(episode, weights)
    word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    bag_of_words(episode, word_sentiments)

    print "max_ent vs gold: %s" % compare_scores(episode, score1="maxent_score", score2="gold_score")
    print "bow vs gold: %s" % compare_scores(episode, "bow_score", score2="gold_score")
