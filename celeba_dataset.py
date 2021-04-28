
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
            l = part_line.split()
            part_map[l[0]] = l[1]

        for anno_line in anno_Lines:
            # l = (file name, dict key, actual name, height)
            l = anno_line.split()
            file_name   = l[0]
            dict_key    = l[1]
            actual_name = l[2]
            height      = float(l[3])
            anno        = [file_name, actual_name, height]

            if part_map[l[0]] == '0':
                self.train_anno.append(anno)
                self.train_img_cnt += 1
            else:
                self.val_anno.append(anno)
                self.val_img_cnt += 1


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


        file_name = anno[idx][0]
        name   = anno[idx][1]
        height = anno[idx][2]
        aod    = 0

        img_path = self._image_dir + '/' + file_name
        # Open the image, convert to RGB
        image = self._transform(Image.open(img_path).convert("RGB"))
        height = torch.FloatTensor([height])



        return {'image': image, 'name': name, 'height': height, 'age_of_death': aod}


def main():

    celeb_set = CelebADataset(image_dir="data/img_align_celeba",
                 annotation_txt_path="data/final_data.txt",
                 data_partition_path="data/list_eval_partition.txt",
                 train_val_flag="train",
                 transform=None)
    loader = DataLoader(celeb_set, batch_size=50, shuffle=False, num_workers=10)
    # print("celeb_set: ", len(loader))
    for batch_id, batch_data in enumerate(loader):
        print(batch_data['height'])

if __name__ == "__main__":
    main()