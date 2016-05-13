"""
This script reads a json file containing episodes of the west wing and saves
them to a pickle file
"""
import json
import re
import pickle

if __name__ == "__main__":
    # wwscripts = json.load(open("data/wwscripts.json"))
    wwscripts = json.load(open("data/wwscripts_trunc.json"))
    new_wwscripts = []
    for episode in wwscripts:
        ''' episode format:
        {'episode': X, 
        'season': X,
        'script': [{ "character": NAME, "text": LINE_STRING }, ... ]
        }
        '''
        new_script = []
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

    pickle.dump(new_wwscripts, open("test_scripts.pkl", "wb"))
