import pandas as pd
import pytz
import json
from datetime import datetime, timedelta
import requests

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
            print(f'Error! Returned status code {r.status_code}')
            print(f"Message: {json['error']['message']}")
            print(f"Reason: {json['error']['reason']}")
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
    

with open("/media/user-1/CameraTraps/NINA/Images/metadata.json") as f:
    metadata = json.load(f)

itr = 0
for datapoint in metadata['annotations']:
    itr += 1
    print(f"{itr}/{len(metadata['annotations'])}")
    if(datapoint['Temperature'] is None and
       datapoint['Latitude'] is not None and
       datapoint['Longitude'] is not None):
       datapoint['Temperature'] = get_temperature(datapoint)

with open("/media/user-1/CameraTraps/NINA/Images/metadata.json", "w") as f:
    json.dump(metadata, f)