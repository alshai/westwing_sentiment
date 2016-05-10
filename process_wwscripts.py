import json
import re
import pickle

if __name__ == "__main__":
    # wwscripts = json.load(open("data/wwscripts.json"))
    wwscripts = json.load(open("data/wwscripts_trunc.json"))
    new_wwscripts = []
    for episode in wwscripts:
        ''' episode format:
        {'episode': X, 'season': X,
        'script': [{ CHARACTER: NAME, text: LINE_STRING }, ... ]
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
                    'text': line[2]})

        new_wwscripts.append(new_script)

pickle.dump(new_wwscripts, open("test_scripts.pkl", "wb"))
