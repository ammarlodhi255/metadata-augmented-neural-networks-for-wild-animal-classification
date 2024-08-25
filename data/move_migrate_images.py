import os
from PIL import Image
import json
import threading

def season_annotation(annotations, bp, lock):
    for anot in annotations:
        process_annotation(anot, bp, lock)

def process_annotation(anot, bp, lock):
    try:
        im = Image.open(bp + anot['image_id'] + ".JPG") # read image

        # extract save path excluding image name
        path = anot['image_id'].split("/")
        path = "/".join(path[:-1])

        # crop and resize
        w, h = im.size
        im = im.crop((0, 0, w, h-100)) # remove metadata band
        im = im.resize((512, 512))

        # Save image
        with lock:
            if not os.path.exists("/media/user-1/CameraTraps/SS_all_samples/" + path):
                os.makedirs("/media/user-1/CameraTraps/SS_all_samples/" + path)
                im.save("/media/user-1/CameraTraps/SS_all_samples/" + anot['image_id'] + ".JPG")
    except: # Corrupt image read (slow drive or bad copy)
        print("Error reading image:", anot['image_id'])

bp = "../../../CameraTraps/snapshotserengeti-unzipped/"
with open(bp + "SnapshotSerengeti_S1-6_v2.1_categories_exists_only.json") as f:
    metadata = json.load(f)

categories = metadata['categories']
annotations = metadata['annotations']

s_annots = [[], [], [], [], [], []]
for anot in annotations:
    s_annots[int(anot['season'][1])-1].append(anot)

threads = []
fs_lock = threading.Lock()

for anot in s_annots:
    # season_annotation(anot, bp, fs_lock)
    t = threading.Thread(target=season_annotation, args=(anot, bp, fs_lock))
    t.start()
    threads.append(t)

for t in threads:
    t.join()