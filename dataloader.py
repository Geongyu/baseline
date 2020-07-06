
import numpy as np
import ipdb
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import ast
import os
import glob
from PIL import Image
import pandas as pd
import random
import torch.utils.data as data

class check_epoch() :
    def __init__(self, epoch=0) :
        self.epoch = epoch

    def give_the_epoch(self, key=None) :
        return self.epoch

def find_file(file_root):
    root_folder = file_root
    sub_folder_image = []
    for folders in root_folder :
        check = os.listdir(folders + "/images")
        for folder in check :
            sub_folder_image.append(folders + "/images/" + str(folder))
    image_file_path = []
    target_file_path = []
    for images in sub_folder_image :
        image_file_list = glob.glob(images + "/*")
        file_list_img = [file for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
        file_list_mask = [file.replace("images", "masks").replace(".png", "_mask.gif") for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
        image_file_path += file_list_img
        target_file_path += file_list_mask

    image_file_path.sort()
    target_file_path.sort()

    return image_file_path, target_file_path

class Classification_Data(Dataset) :
    def __init__(self, exam_root, net="efficientnet", aug=False, smooth=False) :
        '''
        aug == prograsive resize
        smooth = label smoothing
        '''
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.file_root = exam_root
        self.image_path, self.target_path = self.find_file()
        self.network = net
        self.smooth_label = smooth
        self.aug = aug
        self.resize1 = transforms.Resize((1024, 1024))
        self.resize2 = transforms.Resize((800, 800))
        self.resize3 = transforms.Resize((512, 512))
        self.resize4 = transforms.Resize((256, 256))
        self.resize5 = transforms.Resize((128, 128))
        self.resize6 = transforms.Resize((64, 64))
        self.resize7 = transforms.Resize((32, 32))

    def find_file(self):
        root_folder = self.file_root
        sub_folder_image = []
        for folders in root_folder :
            check = os.listdir(folders + "/images")
            for folder in check :
                sub_folder_image.append(folders + "/images/" + str(folder))
        image_file_path = []
        target_file_path = []
        for images in sub_folder_image :
            image_file_list = glob.glob(images + "/*")
            file_list_img = [file for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
            file_list_mask = [file.replace("images", "masks").replace(".png", "_mask.gif") for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
            image_file_path += file_list_img
            target_file_path += file_list_mask

        image_file_path.sort()
        target_file_path.sort()

        return image_file_path, target_file_path

    def __len__(self) :
        #return 1
        return len(self.image_path)

    def give_the_epoch(self, epoch=0) :
        self.epoch = epoch

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        label = Image.open(self.target_path[idx])
        progressive = str(self.aug)
        if progressive == "True" :
            epoch = self.epoch
            if epoch == 0 :
                image = self.resize7(image)
            elif epoch == 1 or epoch == 2 :
                image = self.resize6(image)
            elif epoch == 3 or epoch == 4 or epoch == 5 :
                image = self.resize5(image)
            elif epoch == 6 or epoch == 7 or epoch == 8 or epoch == 9 :
                image = self.resize4(image)
            elif epoch >= 10 :
                image = self.resize3(image)
            else :
                image = image
            #print("Augmentation is Deactivate")
        image = self.transforms1(image)
        image = self.transforms2(image)
        image = image.repeat(3, 1, 1)
        label = np.array(label)
        re_labe = 0
        smoothed = str(self.smooth_label)
        if smoothed == "True" :
            if sum(sum(label)) > 50  :
                label = torch.tensor([0.95]).float()
                re_labe = torch.tensor([1]).float()
            else :
                label = torch.tensor([0.05]).float()
                re_labe = torch.tensor([0]).float()
        else :
            if sum(sum(label)) > 50 :
                label = torch.tensor([1]).float()
                re_labe = torch.tensor([1]).float()
            else :
                label = torch.tensor([0]).float()
                re_labe = torch.tensor([0]).float()
        return image, label, re_labe

class load_kaggle_data(Dataset) :
    def __init__(self, path, label, mode="test") :
        self.file_path = path
        self.mode = mode
        file_list = os.listdir(path)
        label = pd.read_csv(label)
        label = label.sort_values(by=["ID"], axis=0)
        label = label.reset_index(drop=True)
        self.label_file = label
        file_list.sort()
        self.file_list = file_list
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, idx) :
        labels = self.label_file.iloc[idx][1]
        labels = torch.tensor([labels]).float()
        images = Image.open(self.file_path + "/" + self.label_file.iloc[idx][0][:-4] + ".png")

        images = self.transforms1(images)
        images = self.transforms2(images)
        images = images.repeat(3, 1, 1)

        return images, labels

class MTL_Kaggle_Data_with_Balanced(Dataset) :
    def __init__(self, path, label, length) :
        self.file_path = path 
        label = pd.read_csv(label)
        label = label.sort_values(by=["ID"], axis=0)
        label = label.reset_index(drop=True)
        self.label_file = label 
        self.length = length
        file_list = os.listdir(path)  

        self.zero_file = self.label_file.loc[self.label_file["any"] == 0].reset_index(drop=True)
        self.any_file = self.label_file.loc[self.label_file["any"] == 1].reset_index(drop=True)
        file_list.sort()
        self.file_list = file_list
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.image_size = 256
        self.resize = transforms.Resize((256, 256))
    
    def __len__(self) :
        return int(self.length)

    def __getitem__(self, idx) :
        if idx >= (self.length / 2) :
            rdx = np.random.randint(97102, size=1)
            rdx = rdx[0]
            labels = self.any_file.iloc[rdx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.any_file.iloc[rdx].values[0][:-4] + ".png")
            images = self.resize(images)

            images = self.transforms1(images)
            images = self.transforms2(images)
        else :
            rdx = np.random.randint(len(self.zero_file), size=1)
            rdx = rdx[0]
            labels = self.zero_file.iloc[rdx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.zero_file.iloc[rdx][0][:-4] + ".png")
            images = self.resize(images)

            images = self.transforms1(images)
            images = self.transforms2(images)

        return images, labels

class load_kaggle_data_with_balanced(Dataset) :
    def __init__(self, path, label, mode="test") :
        self.file_path = path
        self.mode = mode
        file_list = os.listdir(path)
        label = pd.read_csv(label)
        label = label.sort_values(by=["ID"], axis=0)
        label = label.reset_index(drop=True)
        self.label_file = label
        self.any_file = self.label_file.loc[self.label_file["any"] == 1].reset_index(drop=True)
        file_list.sort()
        self.file_list = file_list
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        return int(len(self.file_list) * 1.5)

    def __getitem__(self, idx) :
        if idx >= len(self.file_list) :
            rdx = np.random.randint(97102, size=1)
            rdx = rdx[0]
            labels = self.any_file.iloc[rdx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.any_file.iloc[rdx].values[0][:-4] + ".png")

            images = self.transforms1(images)
            images = self.transforms2(images)
            images = images.repeat(3, 1, 1)
        else :
            labels = self.label_file.iloc[idx][1]
            labels = torch.tensor([labels]).float()
            images = Image.open(self.file_path + "/" + self.label_file.iloc[idx][0][:-4] + ".png")

            images = self.transforms1(images)
            images = self.transforms2(images)
            images = images.repeat(3, 1, 1)

        return images, labels

class MTL_Self_Supervision(Dataset) :
    def __init__(self, exam_root) :
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5], [0.5])
        self.image_size = 256
        self.mask_size = 64
        self.file_root = exam_root
        self.image_path, self.target_path = self.find_file()
        self.inpainting
        self.resize = transforms.Resize((256, 256))
        

    def find_file(self):
        root_folder = self.file_root
        sub_folder_image = []

        for folders in root_folder :
            check = os.listdir(folders + "/images")
            for folder in check :
                sub_folder_image.append(folders + "/images/" + str(folder))

        image_file_path = []
        target_file_path = []

        for images in sub_folder_image :
            image_file_list = glob.glob(images + "/*")
            file_list_img = [file for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
            file_list_mask = [file.replace("images", "masks").replace(".png", "_mask.gif") for file in image_file_list if file.endswith(".png") or file.endswith(".gif")]
            image_file_path += file_list_img
            target_file_path += file_list_mask

        image_file_path.sort()
        target_file_path.sort()

        return image_file_path, target_file_path

    def __len__(self) :
        return len(self.image_path)

    def give_the_epoch(self, epoch=0) :
        self.epoch = epoch
    
    def inpainting(self, img) :
        i = (self.image_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i 

    def __getitem__(self, idx):

        image = Image.open(self.image_path[idx])
        label = Image.open(self.target_path[idx])
        image = self.resize(image)
        label = self.resize(label)
        label = np.array(label, dtype=np.uint8)
        label = (label != 0) * 1.0
        image = self.transforms1(image)
        image = self.transforms2(image)
        label = self.transforms1(label)

        masked_img, masked_label = self.inpainting(image)

        return image, label, masked_img, masked_label

