import json
from places_v2 import predict
from PIL import Image
import numpy as np
import time

def get_samples(bp):
    with open(bp + "metadata.json") as f:
        metadata = json.load(f)

    categories = metadata['categories']
    annotations = metadata['annotations']


    return categories, annotations



def main():
    bp = "/media/user-1/CameraTraps/NINA/Images/"
    categories, annotations = get_samples(bp)
    itr = 1
    avg_time = [time.time()]
    for anot in annotations:
        im = Image.open(bp + anot['Filename'])
        io, preds, atr = predict(im)
        l = [io]
        l.extend(preds)
        l.extend(atr)
        l = np.array(l).astype(float)
        anot['env_vector'] = list(l)
        if(itr % 100 == 0):
            avg_time.append(time.time())
            time_left = np.array(avg_time)
            temp = time_left[1:] - time_left[:-1]
            remaining = (temp.mean()/100)*(len(annotations) - itr)
            hours, rem = divmod(remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"{itr}/{len(annotations)}, time remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}, average per annotation: {round(temp.mean()/100, 4)}")
            
        itr += 1


    metadata = {'categories': categories, 'annotations': annotations}
    with open(bp + "metadata.json", "w") as f:
        json.dump(metadata, f)

main()