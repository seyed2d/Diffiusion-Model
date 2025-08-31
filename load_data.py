%%writefile load_data.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, image_size=64):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1)  # Scale pixels from [0, 255] to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths) # Return total number of images in the dataset

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), 0  # dummy label

def load_UTKFace(batch_size=4, path='/content/drive/MyDrive/data/crop_part1', image_size=64):
    dataset = SimpleImageDataset(path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader # Loads UTKFace dataset and returns a PyTorch DataLoader with transformed images
