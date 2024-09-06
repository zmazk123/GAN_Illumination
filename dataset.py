import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        dir = os.path.join(self.img_dir, "depth")
        dataset_size = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
        return dataset_size

    def __getitem__(self, idx):
        #print(idx)
        img_path_depth = os.path.join(self.img_dir, "depth", str(idx) + ".png")
        image_depth = read_image(img_path_depth)
        #depth image has only 1 channel, so we copy the same chanell to 3 channels
        if(len(image_depth) == 1):
            image_depth = image_depth.repeat(3, 1, 1)

        image_depth = self.__normalize(image_depth)

        img_path_normal = os.path.join(self.img_dir, "normal", str(idx) + ".png")
        image_normal = read_image(img_path_normal)
        image_normal = self.__normalize(image_normal)

        img_path_diffuse = os.path.join(self.img_dir, "diffuse", str(idx) + ".png")
        image_diffuse = read_image(img_path_diffuse)
        image_diffuse = self.__normalize(image_diffuse)

        img_path_direct = os.path.join(self.img_dir, "direct illumination", str(idx) + ".png")
        image_direct = read_image(img_path_direct)
        image_direct = self.__normalize(image_direct)

        img_path_global = os.path.join(self.img_dir, "global illumination", str(idx) + ".png")
        image_global = read_image(img_path_global)
        image_global = self.__normalize(image_global)

        return image_depth, image_normal, image_diffuse, image_direct, image_global

    def __normalize(self, image):
        # normalize [0,255] -> [-1,1]
        min = image.min()
        max = image.max()
        image = torch.FloatTensor(image.size()).copy_(image)
        image.add_(-min).mul_(1.0 / (max - min))
        image = image.mul_(2).add_(-1)

        return image