import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import time
import os
import requests
from places_v2 import predict
import cv2
from imageio.v2 import imread
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import math
import pandas as pd
import pytz


translation_dict = {
        "rev": "Fox",
        "rådyr": "Deer",
        "grevling": "Weasel",
        "fugl": "Bird",
        "gaupe": "Lynx",
        "hjort": "Deer",
        "elg": "Deer",
        "katt": "Cat",
        "sau": "Sheep",
        "ekorn": "Rodent",
        "mår": "Weasel",
        "rugde": "Bird",
        "hare": "Rabbit",
        "skogshøns": "Bird",
        "svarttrost": "Bird",
        "nøtteskrike": "Bird",
        "kjøttmeis": "Bird",
        "smågnager": "Rodent",
        "storfe": "Cattle",
        "villsvin": "Boar",
        "rovfugl": "Bird",
        "ringdue": "Bird",
        "spettmeis": "Bird",
        "meis sp.": "Bird",
        "måltrost": "Bird",
        "trost sp.": "Bird",
        "storfugl": "Bird",
        "ulv": "Wolf",
        "jerv": "Weasel",
        "bjørn": "Bear",
        "orrfugl": "Bird",
        "kattugle": "Bird",
        "blåmeis": "Bird",
        "oter": "Otter",
        "røyskatt": "Weasel",
        "musvåk": "Bird",
        "jerpe": "Bird",
        "kråke": "Bird",
        "skjære": "Bird",
        "kanadagås": "Bird",
        "trane": "Bird",
        "gråtrost": "Bird",
        "svarthvit fluesnapper": "Bird",
        "dåhjort": "Deer",
        "svartspett": "Bird",
        "flaggspett": "Bird",
        "rødvingetrost": "Bird",
        "dompap": "Bird",
        "sørhare": "Rabbit",
        "snømus": "Rodent",
        "ravn": "Bird",
        "gråhegre": "Bird",
        "bever": "Rodent",
        "bird": "Bird",
        "mink": "Weasel",
        "ilder": "Weasel",
        "bokfink": "Bird",
        "duetrost": "Bird",
        "rødstrupe": "Bird",
        "grønnspett": "Bird",
        "lappugle": "Bird",
        "rein": "Deer",
        "hund": "Dog",
        "piggsvin": "Hedgehog",
        "nøttekråke": "Bird",
        "grønnfink": "Bird",
        "hest": "Horse",
        "vandrefalk": "Bird",
        "lavskrike": "Bird",
        "gråfluesnapper": "Bird"
    }

def combine_classes(c):
    c = c.lower()
    if c in translation_dict.keys():
        return translation_dict[c]
    else:
        return c

def get_nearest_sources(latitude, longitude, client_id):
    sources_endpoint = 'https://frost.met.no/sources/v0.jsonld'
    sources_parameters = {
        'geometry': f'nearest(POINT({longitude} {latitude}))',
        'nearestmaxcount': 5,
    }
    sources_response = requests.get(sources_endpoint, sources_parameters, auth=(client_id, ''))
    sources_json = sources_response.json()
    if sources_response.status_code == 200:
        sources_data = sources_json['data']
        source_ids = [source['id'] for source in sources_data]
        return ','.join(source_ids)
    else:
        return None

def get_temperature(datapoint):
    with open("/home/user-1/prog/masterthesis/src/py/yr_client_id.token") as f:
        key = f.readline().strip()

    latitude = datapoint['Latitude']
    longitude = datapoint['Longitude']
    timestamp = datapoint['Date']
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

    start_time = dt - timedelta(days=1)
    end_time = dt + timedelta(days=1)

    time_range = f"{start_time.isoformat()}/{end_time.isoformat()}"

    client_id = key
    nearest_sources = get_nearest_sources(latitude, longitude, client_id)
    if not nearest_sources:
        print("Error fetching nearest sources.")
        return None

    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': nearest_sources,
        'elements': 'mean(air_temperature P1D)',
        'referencetime': time_range,
    }
    try: 
        r = requests.get(endpoint, parameters, auth=(client_id, ''))
        json = r.json()
        if r.status_code == 200:
            data = json['data']
        else:
            # print(f'Error! Returned status code {r.status_code}')
            # print(f"Message: {json['error']['message']}")
            # print(f"Reason: {json['error']['reason']}")
            return None

        df = pd.DataFrame()
        for i in range(len(data)):
            row = pd.DataFrame(data[i]['observations'])
            row['referenceTime'] = data[i]['referenceTime']
            row['sourceId'] = data[i]['sourceId']
            df = pd.concat((df,row))

        df = df.reset_index()
        df['referenceTime'] = pd.to_datetime(df['referenceTime'])
        # Define your target timestamp
        target_timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        target_timestamp = target_timestamp.replace(tzinfo=pytz.UTC)

        # Calculate the time difference between each timestamp and the target timestamp
        df['time_difference'] = abs(df['referenceTime'] - target_timestamp)

        # Find the minimum time difference
        min_time_difference = df['time_difference'].min()

        # Filter the DataFrame to get the rows with the minimum time difference
        closest_rows = df[df['time_difference'] == min_time_difference]

        # Calculate the average temperature if there are multiple equally closest timestamps
        average_temperature = closest_rows['value'].mean()

        return average_temperature
    except:
        print("Could not fetch data for datapoint:", datapoint["Filename"])
        return None

def clean_entry(d):
    c = ""
    if (
        "Tekst" in d.keys() and d["Filnavn"].split(".")[-1].lower() == "mp4"
    ):  # removes movies from dataset
        return None
    for art in arter:  # find species
        if art["ArtID"] == d["FK_ArtID"]:
            c = art["Navn"]
            break
    c = combine_classes(c)  # combines and tolower() the species

    if "annet" in c or "ukjent" in c:  # remove bad labels (combined or ilegable)
        return None

    datapoint = {}
    datapoint["Date"] = d["Dato"]
    datapoint["Temperature"] = d["Temperatur"]
    datapoint["Camera_Type"] = d["CameraType"]
    datapoint["Camera_Model"] = d["CameraModel"]
    datapoint["Filename"] = d["Filnavn"]
    datapoint["Exposure_Time"] = d["ExposureTime"]
    datapoint["ISO"] = d["ISO"]
    datapoint["Brightness"] = d["Brightness"]
    datapoint["Contrast"] = d["Contrast"]
    datapoint["Sharpness"] = d["Sharpness"]
    datapoint["Saturation"] = d["Saturation"]
    datapoint["Latitude"] = d["latitude"]
    datapoint["Longitude"] = d["longitude"]
    # print(datapoint['Temperature'], datapoint['Latitude'], datapoint['Longitude'], datapoint['Date'])
    if( datapoint['Temperature'] is None and 
        datapoint['Latitude'] is not None and 
        datapoint['Longitude'] is not None):
        
        datapoint['Temperature'] = get_temperature(datapoint)


    datapoint["Species"] = c

    if c not in categories.keys():
        categories[c] = 1
    else:
        categories[c] += 1
    return datapoint

def create_metadata_vectors(entry, entry_im):
    date = entry["Date"]
    try:
        dt = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
        month = dt.month - 1
        day = dt.day - 1
        hour = dt.hour
    except (TypeError, ValueError):
        # Handle None values or invalid datetime format
        month, day, hour = None, None, None

    # One-hot encode datetime components
    if month is not None:
        month_tensor = nn.functional.one_hot(torch.tensor(month), 12)
    else:
        month_tensor = torch.zeros(12)
    if day is not None:
        day_tensor = nn.functional.one_hot(torch.tensor(day), 31)
    else:
        day_tensor = torch.zeros(31)
    if hour is not None:
        hour_tensor = nn.functional.one_hot(torch.tensor(hour), 24)
    else:
        hour_tensor = torch.zeros(24)

    # Concatenate the datetime tensors
    datetime_tensor = torch.cat([month_tensor, day_tensor, hour_tensor]).to(
        torch.float32
    )

    # Environent tensor
    (io_image, probs, responses_attribute) = predict(entry_im)
    io_image, probs, responses_attribute = (
        torch.tensor(io_image),
        torch.tensor(probs),
        torch.tensor(responses_attribute),
    )

    env_tensor = torch.cat([io_image.reshape(1), probs, responses_attribute]).to(
        torch.float32
    )

    entry["datetime_vector"] = datetime_tensor.tolist()
    env_tensor = env_tensor.tolist()
    for i in range(len(env_tensor)):
        env_tensor[i] = round(env_tensor[i], 4)
    entry["env_vector"] = env_tensor

    return entry

def process_image(entry):
    with download_lock:
        im = cv2.imread(image_base_path + entry["Filename"])
    h, w, _ = im.shape
    im = im[
        50 : h - 80, 0:w
    ]  # cropping away metadata from image file (band on top and bottom of image)
    im = cv2.resize(im, (512, 512))
    entry = create_metadata_vectors(entry, Image.fromarray(im))
    with download_lock:
        cv2.imwrite(dst + entry["Filename"], im)
        new_annotations.append(entry)

def download_image(entry):
    with download_lock:
        # wait for at least 1 second since last download before downloading an image
        global t0
        t1 = datetime.now()
        delta = (t1 - t0).total_seconds()
        seconds_to_wait = 0.1
        if delta < seconds_to_wait:
            time.sleep(seconds_to_wait - delta)
        t0 = t1

        # save image
        img_data = requests.get(
            "https://viltkamera.nina.no/Media/" + entry["Filename"]
        ).content
        with open(image_base_path + entry["Filename"], "wb") as f:
            f.write(img_data)

def process_file(file):
    with open(raw_metadata + file) as f:
        metadata = json.load(f)
    
    processed_entries = []
    for entry in metadata:
        entry = clean_entry(entry)
        if entry is not None:
            processed_entries.append(entry)
    return processed_entries

# Lock for the download_image function
download_lock = threading.Lock()

t0 = datetime.now()  # used to bottlecap downloads per second from website
t1 = None

raw_metadata = "/media/user-1/CameraTraps/NINA_raw/raw_metadata/"
image_base_path = "/media/user-1/CameraTraps/NINA_raw/new_images/"
dst = "/media/user-1/CameraTraps/NINA/Images/"

if not os.path.exists(dst):
    os.makedirs(dst)
with open("arter.json") as f:
    arter = json.load(f)

categories = {}
annotations = []
files = os.listdir(raw_metadata)

MAX_WORKERS_PROCESS = 16
with ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as executor:
    futures = {executor.submit(process_file, file): file for file in files}
    progress = tqdm(as_completed(futures), total=len(files), desc="Processing files")
    for future in progress:
        file = futures[future]
        try:
            processed_entries = future.result()
            annotations.extend(processed_entries)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# find majority class
maxNum = 0
for key in categories.keys():
    maxNum = max(maxNum, categories[key])

# remove categories with less than .1% of majority class samples
new_categories = {}
for key in categories.keys():
    if(categories[key] > maxNum//1000):
        new_categories[key] = categories[key]
categories = new_categories

# remove samples no longer among the categories
new_annotations = []
for entry in annotations:
    if(entry['Species'] in categories.keys()):
        new_annotations.append(entry)
annotations = new_annotations

files = os.listdir(image_base_path)

new_annotations = []

# Download all images first
MAX_WORKERS_DOWNLOAD = 8  # Adjust the number of download threads based on your system
download_entries = [entry for entry in annotations if not os.path.isfile(os.path.join(image_base_path, entry['Filename']))]

with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as executor:
    futures = {executor.submit(download_image, entry): entry for entry in download_entries}
    progress = tqdm(as_completed(futures), total=len(download_entries), desc="Downloading images")
    for future in progress:
        entry = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error downloading image {entry['Filename']}: {e}")


# Process images after downloading
MAX_WORKERS_PROCESS = 8  # Adjust the number of processing threads based on your system
process_entries = [entry for entry in annotations ]#if not os.path.isfile(os.path.join(dst, entry['Filename']))]
# TODO: Add entries for existing images in the NINA/Images folder (if you do not process all images)

with ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as executor:
    futures = {executor.submit(process_image, entry): entry for entry in process_entries}
    progress = tqdm(as_completed(futures), total=len(process_entries), desc="Processing images")
    for future in progress:
        entry = futures[future]
        try:
            result = future.result()
            if result is not None:
                new_annotations.append(result)
        except Exception as e:
            print(f"Error processing image {entry['Filename']}: {e}")


classNames = list(categories.keys())
for i in range(len(new_annotations)):
    new_annotations[i]['Species_ID'] = classNames.index(new_annotations[i]['Species'])

metadata = {'categories': categories, 'annotations': new_annotations}

with open(dst + "metadata.json", "w") as f:
    json.dump(metadata, f)
