import os
import numpy as np
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex,ImageTextPaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=True, augment=True):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment,images_list_file=training_images_list_file)
    def __name__(self):
        return str(CustomTrain)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, augment=False):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, augment=augment,images_list_file=test_images_list_file)
    def __name__(self):
        return str(CustomTest)

class CustomTextTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=True, augment=True):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImageTextPaths(paths=paths, size=size, random_crop=random_crop, augment=augment,images_list_file=training_images_list_file,split="train")
    def __name__(self):
        return str(CustomTextTrain)

class CustomTextTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, augment=False):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImageTextPaths(paths=paths, size=size, random_crop=random_crop, augment=augment,images_list_file=test_images_list_file,split="test")

    def __name__(self):
        return str(CustomTextTest)
