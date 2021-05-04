
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import re
from collections import Counter
import csv


sys.path.append(os.getcwd())


class CelebADataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self,
                 image_dir,
                 annotation_txt_path,
                 data_partition_path,
                 train_val_flag="train",
                 transform=None):

        self._image_dir = image_dir
        self._transform = transform if transform != None else transforms.Compose([transforms.ToTensor()])
        self.train_val_flag = train_val_flag
        self.train_anno = []
        self.val_anno = []
        self.train_img_cnt = 0
        self.val_img_cnt = 0

        # Parse annotation file
        annotation_file = open(annotation_txt_path, 'r')
        anno_Lines = annotation_file.readlines()

        # Partition file
        part_file = open(data_partition_path, 'r')
        part_Lines = part_file.readlines()
        part_map = {}

        for part_line in part_Lines:
            line_items = part_line.split()
            part_map[line_items[0]] = line_items[1]

        for anno_line in anno_Lines:
            # line_items = (file name, dict key, actual name, height)
            line_items = anno_line.split()
            file_name   = line_items[0]
            dict_key    = line_items[1]
            actual_name = line_items[2]
            orientation = line_items[3]
            anno        = [file_name, actual_name, orientation]

            if part_map[line_items[0]] == '0':
                self.train_anno.append(anno)
                self.train_img_cnt += 1
            else:
                self.val_anno.append(anno)
                self.val_img_cnt += 1

        # # calculate mean & std of heights in training data
        # heights = np.array([h for _, _, h in self.train_anno])
        # self._height_mean = np.mean(heights)
        # self._height_std = np.std(heights)


    def __len__(self):
        if self.train_val_flag == "train":
            return len(self.train_anno)
        elif self.train_val_flag == "val":
            return len(self.val_anno)

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
            train_val_flag: flag for training or validation data
        Return:
            A dict containing multiple torch tensors for image and other info.
        """

        if self.train_val_flag == "train":
            anno = self.train_anno
        elif self.train_val_flag == "val":
            anno = self.val_anno

        file_name   = anno[idx][0]
        name        = anno[idx][1]
        orientation = anno[idx][2]
        
        if anno[idx][2] == "straight":
            orientation = 0
        else:
            orientation = 1
        
        orientation = torch.LongTensor([orientation])

        img_path = self._image_dir + '/' + file_name
        # Open the image, convert to RGB
        image = self._transform(Image.open(img_path).convert("RGB"))

        return {'image': image, 'name': name, 'orientation': orientation}


def main():

    celeb_set = CelebADataset(image_dir="../data/img_align_celeba",
                 annotation_txt_path="../data/final_gay_data.txt",
                 data_partition_path="../data/list_eval_partition.txt",
                 train_val_flag="train",
                 transform=None)
    loader = DataLoader(celeb_set, batch_size=50, shuffle=False, num_workers=10)
    # print("celeb_set: ", len(loader))
    for batch_id, batch_data in enumerate(loader):
        print(batch_data['orientation'])

if __name__ == "__main__":
    main()