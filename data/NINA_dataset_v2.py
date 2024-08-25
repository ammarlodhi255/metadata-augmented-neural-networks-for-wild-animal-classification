from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from imblearn.over_sampling import BorderlineSMOTE
import random



class NINADataset(Dataset):
    def __init__(self, annotations, base_path, transform=None, augment_transform=None, augment=False):
        self.annotations = annotations
        self.class_seperated_annotations = self.separate_annotations(annotations)
        self.base_path = base_path
        self.transform = transform
        self.augment = augment
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.annotations)

    def separate_annotations(self, annotations):
        class_seperated_annotations = defaultdict(list)
        for anot in annotations:
            temp = []
            temp.extend(anot['datetime_vector'])
            temp.extend(anot['env_vector'])
            class_seperated_annotations[anot['Species']].append(temp)
        
        d = {}
        for key in class_seperated_annotations:
            d[key] = np.array(class_seperated_annotations[key])

        return d

    def __getitem__(self, index):
        image_name = self.annotations[index]['Filename']

        # Load the image
        image = Image.open(self.base_path + image_name)

        # load metadata tensor
        # most relevant attributes, found during analysis of all data
        keep_attr = [93, 74, 89, 87, 88, 86, 40, 70, 83, 81, 27, 42, 94, 91, 99, 44, 77, 82, 62, 101, 69, 6, 8, 26, 32, 76, 38, 7, 75, 54, 36, 33, 95, 41, 59, 73, 100, 45]
        env_vector = self.annotations[index]['env_vector']
        io = env_vector[0]
        place = env_vector[1:366]
        attr = env_vector[366:]
        # most relevant attributes, sorted by importance (index)
        attr = [attr[i] for i in keep_attr]
        
        

        datetime_tensor = torch.tensor(self.annotations[index]['datetime_vector'])
        t = self.annotations[index]['Temperature']
        if(t is None):
            t = [0, 0]
        else:
            t = str(t)
            t = t.replace("C", "")
            t = t.strip()
            if(len(t) < 1): # no temperature given
                t = [0, 0]
            else:
                t = [1, float(t)]
        pos = [float(self.annotations[index]['Latitude']), float(self.annotations[index]['Longitude'])]

        full_tensor = torch.cat([datetime_tensor, torch.tensor(t), torch.tensor(pos)])

        latitude = self.annotations[index]['Latitude']
        if(latitude is None):
            latitude = 0
        longitude = self.annotations[index]['Longitude']
        if(longitude is None):
            longitude = 0
        temp = self.annotations[index]['Temperature']
        if(temp is None):
            temp = 0

        
        # Apply augmentation to the image
        if self.augment:
            image = self.augmentation_pipeline(image)


        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Return the image, metadata tensor, and species ID
        return image, full_tensor, self.annotations[index]['Species_ID']

    def augmentation_pipeline(self, image):
        # augment full_tensor using ndarray datapoints as a reference
        if(self.augment_transform is None):
            raise TypeError("The augment is of type None")
        # Implement your augmentation pipeline here or call a pre-defined pipeline
        return Image.fromarray(self.augment_transform(image=np.array(image))['image'])