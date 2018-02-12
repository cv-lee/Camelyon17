import torch.utils.data as data

import os
import openslide
import cv2
import numpy as np
import csv

from PIL import Image
from multiprocessing import Process
from user_define import config as cf
from user_define import hyperparameter as hp



def make_tissue_mask(): 
    ''' make tissue mask using binary threshold and otsu threshold
        return tissue mask array which has tissue locations

    '''

    slide = openslide.OpenSlide(cf.demo_slide_path)

    tissue_mask = slide.read_region((0,0), hp.mask_level, slide.level_dimensions[hp.mask_level])
    tissue_mask = cv2.cvtColor(np.array(tissue_mask), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)
    return tissue_mask



def make_patch(tissue_mask):
    ''' extract patches in the whole slide using tissue mask
        return patch list and patch location

    Args:
        tissue_mask (numpy array): tissue mask list
    '''

    slide = openslide.OpenSlide(cf.demo_slide_path)

    p_size = hp.patch_size
    width, height = np.array(slide.level_dimensions[0])//304
    total = width * height
    all_cnt, patch_cnt = 0,0
    step = int(p_size/(2**MASK_LV))

    patch_list = []
    patch_location = []
    for i in range(width):
        for j in range(height):
            tissue_mask_sum = tissue_mask[step * j : step * (j+1),
                                            step * i : step * (i+1)].sum()
            tissue_mask_max = step * step * 255
            tissue_area_ratio = tissue_mask_sum / tissue_mask_max

            if tissue_area_ratio > hp.tissue_threshold:
                patch = np.array(slide.read_region((p_size*i, p_size*j),0,(p_size,p_size)))
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
                patch_list.append(patch)
                patch_location += [[i,j]]
                patch_cnt += 1
            
            all_cnt +=1
            print('\rProcess: %.3f%%, All Patch: %d, Tissue Patch: %d' 
                %(100.*all_cnt/total, all_cnt, patch_cnt), end='')

    patch_list = np.array(patch_list)
    patch_location = np.array(patch_location)
    
    return patch_list, patch_location



def get_dimension(lv, patch_size):
    ''' get width and height of level and return them
        
    Args:
        lv (int): level of slide
        patch_size(int): patch size 
    '''

    slide = openslide.OpenSlide(cf.demo_slide_path)
    width, height = np.array(slide.level_dimensions[lv])//patch_size
    return width, height



class dataset_eval(data.Dataset): 
    ''' camelyon17 evaluation dataset class for pytorch datalodaer

    '''

    def __init__(self, patch_list, patch_location, transform=None):
        self.transform = transform
        self.location = patch_location
        self.data = patch_list

    def __getitem__(self,index):
        img = self.data[index]
        img  = Image.fromarray(np.uint8(img))
        
        if self.transform is not None:
            img = self.transform(img)

        return img, self.location[index]

    def __len__(self):
        return len(self.data)



def get_dataset(transform):
    ''' dataset function to get test dataset

    Args:
        transfrom (torchvision.transforms): test set transform for data argumentation
    '''

    tissue_mask = make_tissue_mask()
    patch_list, patch_location = make_patch(tissue_mask)
    test_dataset = dataset_eval(patch_list, patch_location, transform)
    return test_dataset
