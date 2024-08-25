import json, os, sys
from dateutil.parser import parse
import requests
import time


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def main(jsonPath, savePath):
    with open(jsonPath, encoding='utf-8') as fh:
            data = json.load(fh)
    
    d = {}
    itr = 0 # debugging
    for entry in data:
        itr += 1
        fn = entry['Filnavn']
        if(fn.split(".")[-1] != "mp4"): # don't download videos
            # used to check if date is part of animal name
            w2 = entry['Tekst'].split(" ")
            # prune date from animal name if needed
            if(len(w2) > 1 and is_date(w2[-1])):
                animal = " ".join(entry['Tekst'].split(" ")[:-1])
                if(animal.lower() not in d.keys()):
                    d[animal.lower()] = 1
                else:
                    d[animal.lower()] += 1
            else:
                animal = " ".join(entry['Tekst'].split(" ")[:-1])
                if(entry['Tekst'].lower() not in d.keys()):
                    d[entry['Tekst'].lower()] = 1
                else:
                    d[entry['Tekst'].lower()] += 1
            
            # read online image
            img_data = requests.get("https://viltkamera.nina.no/Media/" + fn).content
            
            # make dir if not exists
            if(not os.path.exists(savePath + animal + "/")): 
                os.makedirs(savePath + animal + "/")

            # Save image
            with open(savePath + animal + "/" + fn, 'wb') as handler:
                handler.write(img_data)
            
            time.sleep(5)
            if(itr % (168015//1000) == 0):
                print(f'{itr}/168015')

    d1 = {}
    for key in d.keys():
        if(d[key] >= 25):
            d1[key] = d[key]
    print(d1)
    print(itr)

    

if __name__ == "__main__":
    main("/home/user-1/prog/CameraTraps/NINA/combined.json", "/home/user-1/prog/CameraTraps/NINA/images/")