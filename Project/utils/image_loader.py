import torchvision
import torch
import os

from utils.config import Config

# load images and set label
def load_data():
    config = Config().params
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dir = os.path.join(config["image_folder"], "train")
    vad_dir = os.path.join(config["image_folder"], "vad")

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    vad_set = torchvision.datasets.ImageFolder(root=vad_dir, transform=transform)
    vad_loader = torch.utils.data.DataLoader(vad_set, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    if len(train_set.classes) == len(vad_set.classes):
        config["class_num"] = len(train_set.classes)
    else:
        print("set train and test classes to be same")
    return train_loader, vad_loader
