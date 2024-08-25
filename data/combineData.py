import json, os, sys


def main(path):
    """
    cobines all files in given folder into one new datafile of all json objects

    path(str): path to folder with all json objects
    """
    files = os.listdir(path)
    files.sort()
    if("combined.json" in files):
        files.remove("combined.json") # remove if file already exists
    d = []
    tally = 0
    for file in files:
        with open(path + file, encoding='utf-8') as fh:
            data = json.load(fh)
            d.extend(data)
            tally += len(data)
    with open(path + 'combined.json', 'w') as f:
       json.dump(d, f, indent=4)
    
    print(tally)

if __name__ == "__main__":
    main(sys.argv[1])