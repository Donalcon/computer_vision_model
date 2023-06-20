from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from torchvision import transforms
import torch
import os

transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

class ImageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.class_to_int = {'dublin': 0, 'kerry': 1}
        self.img_names = os.listdir(img_dir)
        self.labels = os.listdir(label_dir)  # this assumes that the labels are in the same order as the images

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        with open(label_path, 'r') as file:
            label = file.read().split()[0].strip()
            label = int(label)

        image = Image.open(img_path).convert('RGB')
        #label = self.class_to_int[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label



train_dataset = ImageDataset("data/dub_vs_kerry/train/images", "data/dub_vs_kerry/train/labels", transform=transform)
val_dataset = ImageDataset("data/dub_vs_kerry/valid/images", "data/dub_vs_kerry/valid/labels", transform=transform)
test_dataset = ImageDataset("data/dub_vs_kerry/test/images", "data/dub_vs_kerry/test/labels", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
