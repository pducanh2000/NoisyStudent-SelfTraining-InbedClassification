import random
import numpy as np 
import cv2


import torch
from torchvision.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor, Normalize


class PmatDataset(Dataset):
    def __init__(self, data, preprocess=None):
        super(PmatDataset, self).__init__()
        self.train_mode = train_mode
        self.images = data["image"]
        self.postures = data["posture"].reshape(-1)

        self.perprocessing = preprocessing
    
    def getitem(self, index):
        if self.preprocessing is not None:
            data_item = self.preprocessing(self.images[index], self.postures[index])
        else:
            dât_item = self.transform(self.images[index], self.postures[index])
    
    def __len__(self):
        return len(self.postures)
    
    @staticmethod
    def transform(image, posture):

        image = ToTensor()(image)
        image = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(image)

        return image, torch.tensor(posture, dtype=torch.long)
