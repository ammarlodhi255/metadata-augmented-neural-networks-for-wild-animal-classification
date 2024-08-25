import torch
import cv2
import torch.nn as nn

class SerengetiDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, base_path, transform=None):
        self.annotations = annotations
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations[index]['image_id']

        # Load the image using OpenCV
        image = cv2.imread(self.base_path + image_name + ".JPG")

        # Apply transformations to the image and targets
        if self.transform:
            image = self.transform(image)
        
        # convert datetime to usable continious format
        datetime = self.annotations[index]['datetime']
        d, t = datetime.split(" ")
        d = d.split("-")[1:]
        d1 = nn.functional.one_hot(torch.tensor(int(d[0])-1), 12)
        d2 = nn.functional.one_hot(torch.tensor(int(d[1])-1), 31)
        
        t = int(t.split(":")[0])
        t = nn.functional.one_hot(torch.tensor(t), 24)

        return image, torch.cat([d1, d2, t]).to(torch.float32), self.annotations[index]['category_id']