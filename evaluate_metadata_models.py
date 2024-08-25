import numpy as np
import os
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 22}

matplotlib.rc('font', **font)

class_names = [
    'Fox',
    'Deer',
    'Weasel',
    'Bird',
    'Lynx',
    'Cat',
    'Sheep',
    'Squirrel',
    'Rabbit',
    'Rodent',
    'Cattle',
    'Boar',
    'Wolf',
    'Bear'
 ]

feature_names = ['Datetime', 'Places', 'Position & Temperature','Scene features']

def printres():
    for i in range(4):
        feature_includes = []
        for j in range(len(best_features[i])):
            feature_includes.append(feature_names.index(best_features[i][j]))
        temp = str(best_features[i])[1:-1].replace("Datetime", "DT").replace("Places", "Pl").replace("Position & Temperature", "P & T").replace("Scene features", "SA").replace("&", "\\&")
        class_includes = []
        for j in range(len(best_classes[i])):
            class_includes.append(class_names.index(best_classes[i][j]))
        typestr = f"\\textit{{ {str(class_includes)[1:-1]} }} & \\textit{{ {temp} }} & {best_acc[i]} & {best_fpr[i]} & {best_fnr[i]} & {best_kappa[i]} \\\\"
        new_str = ""
        special = True
        for char in typestr:
            if(char == "'"):
                if(special):
                    char = "`"
                special = not special
            new_str += char
        print(new_str)
        print("\\hline")
    print("\\hline")


conf_mats = [file for file in os.listdir("./Metadata_only") if("confusion" in file)]
evaluations = [file for file in os.listdir("Metadata_only") if("evaluation" in file)] 
conf_mats = sorted(conf_mats, key=lambda x: int(x.split('_')[-1].split('.')[0]))
evaluations = sorted(evaluations, key=lambda x: int(x.split('_')[-1].split('.')[0]))

best_kappa = [-1, -1, -1, -1]
best_acc = [-1, -1, -1, -1]
best_fpr = [-1, -1, -1, -1]
best_fnr = [-1, -1, -1, -1]
best_features = [None, None, None, None]
best_classes = [None, None, None, None]
cur_classes = 2

animals_old = []
feature_tracker = {}

for file in evaluations:
    with open("./Metadata_only/" + file) as f:
        try:
            # fetching relevant data from evaluation_x.txt
            s = f.readline().strip()
            classes = ast.literal_eval(s.split(":")[-1][1:])                     # Create a list of classes for this evaluation.txt file
            features = ast.literal_eval(s.split(":")[2][1:].split("]")[0] + "]") # super ugly, but gets out the features used
            old = ""
            while(len(s) > 0):
                old = s
                s = f.readline().strip()
            scores = list(old.split("\t"))
            new_scores = []
            for score in scores:
                new_scores.append(float(score))
            scores = new_scores
            kappa = scores[3]
        except:
            print(file)
            for line in f:
                print(line)
            exit()

    if("Cat" in classes): # should not have included cat i think, very small samplesdet
        continue

    if(animals_old != classes):
        animals_old = classes
        feature_tracker[str(animals_old)] = {}
    
    temp = feature_tracker[str(animals_old)]
    if(len(features) not in temp.keys()):
        temp[len(features)] = {"kappa": kappa, "features": features}
    elif(temp[len(features)]['kappa'] < kappa):
        temp[len(features)]['kappa'] = kappa
        temp[len(features)]['features'] = features

    feature_tracker[str(animals_old)] = temp

    if(len(classes) > cur_classes):
        printres()

        cur_classes = len(classes)
        best_kappa = [-1, -1, -1, -1]
        best_features = [None, None, None, None]
        best_classes = [None, None, None, None]

    if(kappa > best_kappa[len(features)-1]):
        best_kappa[len(features)-1] = kappa
        best_acc[len(features)-1] = scores[0]
        best_fpr[len(features)-1] = scores[1]
        best_fnr[len(features)-1] = scores[2]
        best_features[len(features)-1] = features
        best_classes[len(features)-1] = classes

# printing the last class combination (skipped in loop due to the logic)
printres()


d = {}
for k1 in feature_tracker.keys():
    for k2 in feature_tracker[k1].keys():
        if(len(feature_tracker[k1][k2]['features']) not in d.keys()):
            d[len(feature_tracker[k1][k2]['features'])] = {}
        if(str(feature_tracker[k1][k2]['features']) not in d[len(feature_tracker[k1][k2]['features'])].keys()):
            d[len(feature_tracker[k1][k2]['features'])][str(feature_tracker[k1][k2]['features'])] = 0
        d[len(feature_tracker[k1][k2]['features'])][str(feature_tracker[k1][k2]['features'])] += 1


for key in d.keys():
    bars = d[key].keys()
    new_bars = []
    for bar in bars:
        new_bars.append(bar[1:-1].replace("Datetime", "DT").replace("Places", "Pl").replace("Position & Temperature", "P & T").replace("Scene features", "SA"))
    bars = new_bars

    plt.figure(figsize=(12, 7))
    plt.title(f"Best {key} feature(s)")
    # plt.xkcd()
    plt.bar(bars, d[key].values())
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"./best_{key}_features.png")
    # plt.show()