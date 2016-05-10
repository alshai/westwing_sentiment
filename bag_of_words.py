from data_processing import parse_NRC
import pickle


unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"

def build_vocabulary(data):
    # takes in a dictionary of data
    # returns a set of words
    return set(data.keys())

def preprocess_text(episode, vocabulary):
    script = episode["script"]
    new_episode = {
            "episode": episode["episode"],
            "season": episode["season"],
            "script": []
            }
    for line in script:
        character = line["character"]
        text = line["text"] # a string
        new_text = text.split(" ") # a list
        for i, word in enumerate(new_text):
            if word not in vocabulary:
                new_text[i] = unknown_token
        new_line = {
                "character": character,
                "text": new_text
                }
        new_episode["script"].append(new_line)

    return new_episode
        

def bag_of_words(episode, word_sentiments):
    script = episode["script"]
    for i, line in enumerate(script):
        score = 0.0
        for word in line['text']:
            if word_sentiments[word] == "negative":
                score += 0
            elif word_sentiments[word] == "positive":
                score += 1
            elif word_sentiments[word] == "neutral":
                score += 0.5
        score = score / len(line['text'])
        episode['script'][i]['bow_score'] = score

def character_scores(episode, character, score_type='bow_score'):
    script = episode['script']
    for line in script:
        if line["character"] == character:
            print line[score_type]


if __name__ == "__main__":
    episodes = pickle.load(open("test_scripts.pkl", "rb"))
    word_sentiments = parse_NRC("data/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
    word_sentiments[unknown_token] = "neutral"
    vocabulary = build_vocabulary(word_sentiments)

    preprocessed_episodes = []
    for episode in episodes:
        preprocessed_episodes.append(preprocess_text(episode, vocabulary))
    bag_of_words(preprocessed_episodes[0], word_sentiments)
    print character_scores(preprocessed_episodes[0],
            "TOBY", "bow_score")

