from places import predict
import json
from PIL import Image
import random
import os
import numpy as np

def get_samples(bp):
    with open(bp + "metadata.json") as f:
        metadata = json.load(f)

    categories = metadata['categories']
    annotations = metadata['annotations']


    return categories, annotations

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'places/categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'places/IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'places/labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'places/W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def main():
    #bp = "C:/Users/Aslak/prog/uni/Cameratraps/NINA/processed_images/"
    #categories, annotations = get_samples(bp)
    #r1 = random.randint(0, len(annotations)-1)
    #r2 = random.randint(0, len(annotations)-1)
    #fn1 = annotations[r1]['Filename']
    #fn2 = annotations[r2]['Filename']
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    im1 = Image.open("C:/Users/Aslak/prog/uni/masterthesis/src/notebooks/test.jpg")
    im2 = Image.open("C:/Users/Aslak/prog/uni/masterthesis/src/notebooks/test2.jpg")

    (io_image1, probs1, responses_attribute1, idx_a1, CAMs1) = predict(im1)
    (io_image2, probs2, responses_attribute2, idx_a2, CAMs2) = predict(im2)

    print(idx_a2)

main()