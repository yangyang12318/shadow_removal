import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import torch
class ImageDataset(Dataset):
    def __init__(self, root, transforms_1=None, transforms_2=None,transforms_3=None):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.transform3=transforms.Compose(transforms_3)
        self.files_A = sorted(glob.glob(os.path.join(root,  'image') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root,  'mk') + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, 'nmk') + '/*.*'))
        self.files_D = sorted(glob.glob(os.path.join(root, 'nsd') + '/*.*'))
        self.files_E = sorted(glob.glob(os.path.join(root, 'sd') + '/*.*'))
        self.files_F = sorted(glob.glob(os.path.join(root, 'fake') + '/*.*'))
        self.files_G=sorted(glob.glob(os.path.join(root, 'res') + '/*.*'))

    def __getitem__(self, index):

        item_A = self.transform1(Image.open(self.files_A[index % len(self.files_A)]))

        item_B = self.transform2(Image.open(self.files_B[index % len(self.files_B)]))

        #item_B1 = self.transform3(Image.open(self.files_B[index % len(self.files_B)]))
        item_C = self.transform2(Image.open(self.files_C[index % len(self.files_C)]))

        #item_C1 = self.transform3(Image.open(self.files_C[index % len(self.files_C)]))
        item_D = self.transform1(Image.open(self.files_D[index % len(self.files_D)]))

        item_E = self.transform1(Image.open(self.files_E[index % len(self.files_E)]))

        fake_F = self.transform1(Image.open(self.files_F[index % len(self.files_F)]))

        item_G = self.transform2(Image.open(self.files_G[index % len(self.files_G)]))

        return {'A':item_A,'B': item_B,'C':item_C, 'D':item_D,'E':item_E,'F':fake_F,'G':item_G}


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
