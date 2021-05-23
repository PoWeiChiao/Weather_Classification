import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WeatherDataset(Dataset):
    def __init__(self, data_dir, label_set, image_transforms):
        self.data_dir = data_dir
        self.label_set = label_set
        self.image_transforms = image_transforms

        self.images_list = []
        for label in label_set:
            self.images_list.extend(glob.glob(os.path.join(data_dir, label, '*.jpg')))
        
        self.images_list.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.image_transforms(image)

        label = 0
        for i, l in enumerate(self.label_set):
            if l in os.path.basename(self.images_list[idx]):
                label = i
        label = torch.as_tensor(label, dtype=torch.long)

        return image, label

def main():
    data_dir = 'data/train'
    label_set = ['cloudy', 'rain', 'shine', 'sunrise']
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = WeatherDataset(data_dir, label_set, image_transforms)
    print(dataset.__len__())

if __name__ == '__main__':
    main()
        
