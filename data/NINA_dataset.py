from PIL import Image
import torch
from torch.utils.data import Dataset

class NINADataset(Dataset):
    def __init__(self, annotations, base_path, transform=None, augment_transform=None, augment=False):
        self.annotations = annotations
        self.base_path = base_path
        self.transform = transform
        self.augment = augment
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations[index]['Filename']

        # Load the image
        image = Image.open(self.base_path + image_name)

        # load metadata tensor
        env_tensor = torch.tensor(self.annotations[index]['env_vector'])
        datetime_tensor = torch.tensor(self.annotations[index]['datetime_vector'])
        full_tensor = torch.cat([datetime_tensor, env_tensor])
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            self.augmentation_pipeline(image)
        
        # Return the image, metadata tensor, and species ID
        return image, full_tensor, self.annotations[index]['Species_ID']

    def augmentation_pipeline(self, image):
        if(self.augment_transform is None):
            raise TypeError("The augment is of type None")
        # Implement your augmentation pipeline here or call a pre-defined pipeline
        return self.augment_transform(image)