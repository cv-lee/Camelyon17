import os
import os.path
import errno
import numpy as np
import sys
import torch.utils.data as data
import csv
import cv2
import random
sys.path.append('..')

from PIL import Image
from user_define import config as cf


class camel(data.Dataset):
    """

    """
    
    def __init__(self, root, usage='train', limit=0, train_ratio = 0,
                 transform=None, target_transform=None):
        super(camel,self).__init__()
        self.root = root 
        self.transform = transform
        self.target_transform = target_transform
        self.usage = usage # train,val,subtest,test,mining
        self.data = []
        self.labels = []
        self.limit = limit

        if self.usage =='train':
            self.img_name_list = os.listdir(self.root)            
            csv_path = self.root+'label/train_label.csv'

        elif self.usage == 'val':
            csv_path = self.root+'label/valid_label.csv'

        elif self.usage == 'subtest':
            csv_path = self.root+'label/test_label.csv'
            
        elif self.usage == 'mining':
            self.ratio = train_ratio
            csv_path = self.root+'label/mining_label.csv'
            csv_path2 = cf.dataset_path+'train/label/train_label.csv'
        
        else: # self.usage == 'test'
            pass

        if self.usage == 'test':
            for img in os.listdir(self.root):
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array)
        
        else:
            csv_file = open(csv_path, 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)
            cnt = 0
            
            for img, label in csv_reader:
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                if self.usage == 'mining':
                    self.data.append(array)
                    self.data.append(array)
                    self.labels.append(label)
                    self.labels.append(label)
                    cnt -= 1
                cnt += 1
                if cnt > self.limit:
                    break

        if self.usage == 'mining':

            self.limit = len(self.data) * self.ratio
            csv_file = open(csv_path2, 'r', encoding ='utf-8')
            csv_reader = csv.reader(csv_file)
            rat = len(os.listdir(cf.dataset_path+'train')) / self.limit
            
            for img, label in csv_reader:
                rand = random.randint(1,ratio)
                if rand % rat == 0:
                    array = cv2.imread(cf.dataset_path + 'train/' + img, cv2.IMREAD_COLOR)
                    self.data.append(array)
                    self.labels.append(label)
                    cnt += 1
                    if cnt > self.limit:
                        break

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img = self.data[index]
        img = Image.fromarray(img)

        if self.usage == 'train':
            target, filename = self.labels[index], self.img_name_list[index]
        
        elif self.usage == 'val' or self.usage == 'subtest' or self.usage =='mining':
            target = self.labels[index]
        
        else: #self.usage == 'test'
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.usage =='train':
            return img, target, filename
        
        elif self.usage == 'val' or self.usage == 'subtest' or self.usage == 'mining':
            return img, target
        
        else: #self.usage == 'test'
            return img


    def __len__(self):
        return len(self.data)


       
def get_dataset(train_transform, test_transform, train_max, 
                val_max, subtest_max, ratio=0, mining_mode=False):
    train_dataset = camel(cf.dataset_path + 'train/', usage='train',
                            limit = train_max, transform=train_transform)
    
    val_dataset = camel(cf.dataset_path + 'validation/', usage='val',
                            limit = val_max, transform=test_transform)
    
    subtest_dataset = camel(cf.dataset_path + 'test/', usage='subtest', 
                            limit = subtest_max, transform=test_transform)
    
    test_dataset = camel(cf.test_path, usage ='test',transform=test_transform)
    
    if mining_mode == True:
        mining_dataset = camel(cf.dataset_path + 'mining/', usage='mining',
                                train_ratio = ratio, transform=train_transform)
        return train_dataset, val_dataset, subtest_dataset, test_dataset, mining_dataset

    else:
        return train_dataset, val_dataset, subtest_dataset, test_dataset
