import os
import os.path
import errno
import numpy as np
import sys
import torch.utils.data as data
import csv
import cv2
import random

from PIL import Image

DATASET_PATH = '/mnt/disk3/interns/dataset2/' 
TEST_PATH = '/home/interns/test/test0201/' 

TRAIN_NUM = 186800  # Max: 186,800
VAL_NUM = 62940     # Max: 62,940
SUBTEST_NUM = 62940 # Max: 62,940
TRAIN_RAT = 1


class camel(data.Dataset):
    """

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    
    def __init__(self, root, usage='train',
                 transform=None, target_transform=None):
        
        self.root = root 
        self.transform = transform
        self.target_transform = target_transform
        self.usage = usage # train,val,subtest,test,mining
        self.data = []
        self.labels = []

        if self.usage == 'train':
            self.img_name_list = os.listdir(self.root)
            
            csv_file = open(self.root+'label/train_label.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)
            
            cnt = 0
            for img, label in csv_reader:
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array) #np.array로 바꿔주면 더 속도 빨라질 수 도 있음 !!!! 확인해보기
                self.labels.append(label)
                cnt += 1
                if cnt > TRAIN_NUM:
                    break

        elif self.usage == 'val':
            csv_file = open(self.root + 'label/valid_label.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)

            cnt = 0
            for img, label in csv_reader:
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                cnt += 1
                if cnt > VAL_NUM:
                    break

        elif self.usage == 'subtest':
            csv_file = open(self.root+'label/test_label.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)

            cnt = 0
            for img, label in csv_reader:
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array)
                self.labels.append(label)
                cnt += 1
                if cnt > SUBTEST_NUM:
                    break

        elif self.usage == 'test':
            for img in os.listdir(self.root):  
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                self.data.append(array)
        
        else: # self.usage = 'mining'
            # mining dataset
            csv_file = open(self.root+'label/mining_label.csv', 'r', encoding='utf-8')
            csv_reader = csv.reader(csv_file)

            for img, label in csv_reader:
                array = cv2.imread(self.root + img, cv2.IMREAD_COLOR)
                # mining data duplicate 3times
                '''
                    이론적으로 문제없을지 확인해보기
                '''
                self.data.append(array)
                self.data.append(array)
                self.data.append(array)
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
            
            # train dataset
            train_max = len(self.data) * TRAIN_RAT
            cnt = 0

            if train_max > 250000:
                print('TRAIN_ RAT is too high')
                return

            csv_file = open(DATASET_PATH + 'train/label/train_label.csv', 'r', encoding ='utf-8')
            csv_reader = csv.reader(csv_file)
            '''
                이렇게되면 계속 train 폴더내 맨 앞얘들만 사용하므로 랜덤하게 수정필요
                -> 임시방편으로 이렇게 막고 돌리긴함
            '''
            for img, label in csv_reader:
                rand = random.randint(1,5)
                if rand % 5 == 0:
                    array = cv2.imread(DATASET_PATH + 'train/' + img, cv2.IMREAD_COLOR)
                    self.data.append(array)
                    self.labels.append(label)
                    cnt += 1
                    if cnt > train_max:
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
        if self.usage == 'train':
            img, target, filename = self.data[index], self.labels[index], self.img_name_list[index]
        elif self.usage == 'val' or self.usage == 'subtest' or self.usage =='mining':
            img, target = self.data[index], self.labels[index]
        
        else: # self.usage = 'test'
            img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.usage =='train':
            return img, target, filename
        
        elif self.usage == 'val' or self.usage == 'subtest' or self.usage == 'mining':
            return img, target
        
        else: #self.usage = 'test'
            return img


    def __len__(self):
        return len(self.data)

       
def get_dataset(train_transform, test_transform, mining_mode):
    train_dataset = camel(DATASET_PATH + 'train/',
                            usage='train',
                            transform=train_transform)
    
    val_dataset = camel(DATASET_PATH + 'validation/',
                          usage='val', 
                          transform=test_transform)
    
    subtest_dataset = camel(DATASET_PATH + 'test/',
                           usage='subtest', 
                           transform=test_transform)
    
    test_dataset = camel(TEST_PATH,
                            usage ='test',
                            transform=test_transform)
    
    if mining_mode == True:
        mining_dataset = camel(DATASET_PATH + 'mining/',
                                usage='mining',
                                transform=train_transform)
        return train_dataset, val_dataset, subtest_dataset, test_dataset, mining_dataset

    else:
        return train_dataset, val_dataset, subtest_dataset, test_dataset


# For Test
#if __name__ == "__main__":
#    print("check") 
#    camel_dataset = camel('/mnt/disk3/interns/dataset/train/')
