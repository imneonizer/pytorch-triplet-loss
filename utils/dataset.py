import torch
import os, glob
from PIL import Image, UnidentifiedImageError
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from .common import natural_sort


class SimDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = glob.glob(f"{root_dir}/*/*")
        self.labels = [os.path.basename(os.path.dirname(x)).strip() for x in self.images]
        
        self.classes = list(set(self.labels))
        
        self.label_encode = {self.classes[i]:i for i in range(len(self.classes))}
        self.label_decode = {i:self.classes[i] for i in range(len(self.classes))}
        
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
            label = self.label_encode[self.labels[idx]]
        except Exception:
            print("UnidentifiedImageError: ", self.images[idx])
            
            # Recursively load next image
            return self.__getitem__(max(idx+1, len(self.images)-1))

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(label))


def build_dataloader(batch_size, root_dir, transform, shuffle=False, num_workers=0):
    dataset = SimDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader