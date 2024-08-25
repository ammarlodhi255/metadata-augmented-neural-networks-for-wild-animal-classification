import json

def validate_bbox_length(json_file):
    with open(json_file) as f:
        data = json.load(f)
    for annotation in data["annotations"]:
        if len(annotation["bbox"]) != 4:
            return False
    return True

if __name__ == "__main__":
    json_file = "D:\SnapshotSerengetti\images\SS\SnapshotSerengetiBboxes_20190903.json"
    result = validate_bbox_length(json_file)
    if result:
        print("All 'bbox' entries have 4 values.")
    else:
        print("Not all 'bbox' entries have 4 values.")