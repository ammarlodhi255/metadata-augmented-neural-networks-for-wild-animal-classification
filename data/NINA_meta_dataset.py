from PIL import Image
import torch

class NINAMetaDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, base_path, transform=None):
        self.annotations = annotations
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # load metadata tensor
        env_tensor = torch.tensor(self.annotations[index]['env_vector'])
        datetime_tensor = torch.tensor(self.annotations[index]['datetime_vector'])
        full_tensor = torch.cat([datetime_tensor, env_tensor])

        # Return the image, metadata tensor, and species ID
        return full_tensor, full_tensor, self.annotations[index]['Species_ID']
