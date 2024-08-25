import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from collections import defaultdict
import random
from PIL import Image
import matplotlib.cm as cm
from umap.umap_ import UMAP as UMAP

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
img_path = "C:/Users/Aslak/prog/uni/Cameratraps/NINA/processed_images/"
with open(img_path + "metadata.json") as f:
    metadata = json.load(f)

translation_dict = {
    "hare": "Rabbit",
    'rã¥dyr': "Deer",
    'hjortdyr': "Deer",
    "rev": "Fox",
    "ekorn": "Squirrel",
    "gaupe": "Lynx",
    "grevling": "Weasel",
    "mårdyr": "Weasel",
    "mår": "Weasel",
    'mã¥r': "Weasel",
    "fugl": "Bird",
    "Ulv": "Wolf",
    "katt": "Cat",
    'smã¥gnager': "Rodent", # maybe inaccurate
    'skogshã¸ns': "Bird",
    'nã¸tteskrike': "Bird",
    'bjã¸rn': "Bear",
    'villsvin': "Boar",
    'sau': "Sheep",
    'skjã¦re': "Bird",
    'ulv': "Wolf",
    'storfe': "Cattle"
}


def load_imgs(metadata):
    d = defaultdict(list)

    for entry in metadata['annotations']:
        s = entry['Species']
        if s in translation_dict:
            s = translation_dict[s]
        entry['Species'] = s
        d[entry['Species']].append(entry)
    keep = []
    for key in d.keys():
        keep.extend(random.sample(d[key], 100))
    print(d.keys(), len(d.keys()))
    imgs = []
    ids = []
    classes = list(d.keys())
    for entry in keep:
        im = Image.open(img_path + entry['Filename'])
        im = im.resize((128, 128))
        imgs.append(np.array(im).ravel())
        ids.append(classes.index(entry['Species']))
    
    return {"Labels": keep, "Data": np.array(imgs), "Targets": np.array(ids)}


data = load_imgs(metadata)

reducer = UMAP(random_state=42, n_components=2)
reducer.fit(data['Data'])
embedding = reducer.transform(data['Data'])

# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))

plt.scatter(embedding[:, 0], embedding[:, 1], c=data['Targets'], cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(15)-0.5).set_ticks(np.arange(14))
plt.title('UMAP projection of the Viltkamera dataset', fontsize=24)

reducer = UMAP(random_state=42, n_components=3)
reducer.fit(data['Data'])
embedding = reducer.transform(data['Data'])
assert(np.all(embedding == reducer.embedding_))

fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    embedding[:, 2],
    c=data['Targets'], 
    cmap='Spectral', 
    s=5)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.gca().set_aspect('equal', 'datalim')

# Create a ScalarMappable object and use it to create the colorbar
sm = cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=data['Targets'].min(), vmax=data['Targets'].max()))
sm._A = []  # Required for matplotlib < 3.4.0
cbar = plt.colorbar(sm, ax=ax, boundaries=np.arange(15)-0.5)
cbar.set_ticks(np.arange(14))

plt.title('UMAP projection of the Viltkamera dataset', fontsize=24)

plt.show()