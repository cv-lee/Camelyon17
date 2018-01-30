import openslide
import cv2
import numpy as np
import os
import random

from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process

XML_PATH = '/home/interns/annotation/'
SLIDE_PATH = '/mnt/disk3/interns/slide/'
PATCH_PATH = '/home/interns/patch_test/' # Use patch_test file for TESTING

#GLOBAL VARIABLES should move to 'main.py'
MASK_LEVEL = 4 # 4 is fixed 
MAP_LEVEL = 4 # must be same with MASK_LEVLE
SLIDE_NUM = 10

NORMAL_RATIO = 0.5
TUMOR_RATIO = 0.3



def make_dir(slide_num, flags):
    
    '''
        make directory of files using flags

        if flags is tumor_patch or normal patch
        additional directory handling is needed
    '''

    if flags == 'slide':
        return SLIDE_PATH + 'b_' + str(slide_num) + '.tif'

    elif flags == 'xml':
        return XML_PATH + 'b_' + str(slide_num) +'.xml'

    elif flags == 'mask':
        return PATCH_PATH + 'b' + str(slide_num) + '/mask'

    elif flags == 'map':
        return PATCH_PATH + 'b' + str(slide_num) + '/b' + str(slide_num) + '_map.png'

    elif flags == 'tumor_mask':
        return PATCH_PATH + 'b' + str(slide_num) + '/mask/b' + str(slide_num) + '_tumor_mask.png'

    elif flags == 'tumor_patch':
        return PATCH_PATH + 'b' + str(slide_num) + '/tumor/b' + str(slide_num) + '_'

    elif flags == 'normal_mask':
        return PATCH_PATH + 'b' + str(slide_num) + '/mask/b' + str(slide_num) + '_normal_mask.png'

    elif flags == 'normal_patch':
        return PATCH_PATH + 'b' + str(slide_num) + '/normal/b' + str(slide_num) + '_'
    
    elif flags == 'tissue_mask':
        return PATCH_PATH + 'b' + str(slide_num) + '/mask/b' + str(slide_num) + '_tissue_mask.png'

    else:
        print('make_dir flags error')
        return



def chk_file(filedir,filename):
    
    '''
        check whether file(filename) is existed in filedir or not
        if existed, return True / else, return False
    '''

    exist = False

    os.chdir(filedir)
    cwd = os.getcwd()
    
    for file_name in os.listdir(cwd):
        if file_name == filename:
            exist = True
    
    os.chdir('/home/interns/camelyon17')# it can be changed

    return exist



def read_xml(slide_num, mask_level):

    '''
        read xml files which has tumor coordinates list
        return coordinates of tumor areas
    '''

    path = make_dir(slide_num, 'xml') 
    xml = parse(path).getroot()
    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X'))/(2**mask_level)),
                            round(float(area.get('Y'))/(2**mask_level))])
        coors_list.append(coors)
        coors=[]
    return np.array(coors_list)


def make_tumor_patch(slide_num, mask_level):
    
    '''
        make tumor patch using tumor mask
    '''

    # path setting
    slide_path = make_dir(slide_num, 'slide')
    map_path = make_dir(slide_num, 'map')
    mask_folder_path = make_dir(slide_num, 'mask')
    tumor_mask_path = make_dir(slide_num, 'tumor_mask')
    tumor_patch_path = make_dir(slide_num, 'tumor_patch')

    # slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[MAP_LEVEL]))

    # xml loading
    coors_list = read_xml(slide_num, mask_level)

    # tumor mask init
    tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1])
    print('Tumor mask size: (%d,%d)' %(tumor_mask.shape[::-1]))
    
    # check tumor mask / draw tumor mask
    tumor_mask_exist = chk_file(mask_folder_path, 'b' + str(slide_num) + '_tumor_mask.png')
    if tumor_mask_exist == False:
        for coors in coors_list:
            cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)
        cv2.imwrite(tumor_mask_path, tumor_mask)
    else:
        tumor_mask = cv2.imread(tumor_mask_path, 0)

    # draw boundary of tumor in map
    for coors in coors_list:
        cv2.drawContours(slide_map, np.array([coors]), -1, 255, 1)
    
    # parameters for patch init 
    width,height = np.array(slide.level_dimensions[0])//304
    total = width*height
    all_cnt = 0
    patch_cnt = 0
    step = int(304/(2**mask_level))

    # extract tumor patch / draw map
    for i in range(width):
        for j in range(height):
            tumor_mask_sum = tumor_mask[step * j : step * (j+1),
                                        step * i : step * (i+1)].sum()
            tumor_mask_max = step * step * 255
            ratio = tumor_mask_sum / tumor_mask_max

            if ratio > TUMOR_RATIO:
                tumor_patch_name = tumor_patch_path + str(i) + '_' + str(j) + '_.png'
                patch = slide.read_region((304*i,304*j), 0, (304,304))
                patch.save(tumor_patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,0,255), 1)
                patch_cnt += 1
            all_cnt += 1

            print('\rProcess: %.3f%%,  All: %d, Patch: %d'
                    %((100.*all_cnt/total), all_cnt, patch_cnt), end="")
    
    # save map
    cv2.imwrite(map_path, slide_map)



def make_normal_patch(slide_num, mask_level):
    '''
        make normal patch using tissue mask and normal mask
    '''

    # path setting
    slide_path = make_dir(slide_num,'slide')
    mask_folder_path = make_dir(slide_num, 'mask')
    tissue_mask_path = make_dir(slide_num, 'tissue_mask')
    normal_mask_path = make_dir(slide_num, 'normal_mask')
    tumor_mask_path = make_dir(slide_num, 'tumor_mask')
    normal_patch_path = make_dir(slide_num, 'normal_patch')
    map_path = make_dir(slide_num, 'map')
    
    # slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(cv2.imread(map_path, -1))

    # check tissue mask / draw tissue mask
    tissue_mask_exist = chk_file(mask_folder_path, 'b' + str(slide_num) + '_tissue_mask.png')
    if tissue_mask_exist == False:
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _,tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(tissue_mask_path, np.array(tissue_mask))
    else:
        tissue_mask = cv2.imread(tissue_mask_path, 0)
    
    # check normal mask
    normal_mask_exist = chk_file(mask_folder_path, 'b' + str(slide_num) + '_normal_mask.png')
    if normal_mask_exist == False:
        tumor_mask = cv2.imread(tumor_mask_path, 0) 
        height, width = np.array(tumor_mask).shape
        for i in range(width):
            for j in range(height):
                if tumor_mask[j][i] > 127:
                    tissue_mask[j][i] = 0
        normal_mask = np.array(tissue_mask)
        cv2.imwrite(normal_mask_path, normal_mask)
    else:
        normal_mask = cv2.imread(normal_mask_path, 0)

    # parameters for patch init
    width, height = np.array(slide.level_dimensions[0])//304
    total = width * height 
    all_cnt = 0
    patch_cnt = 0
    step = int(304/(2**mask_level))

    # extract normal patch / draw map
    for i in range(width):
        for j in range(height):
            random_num = random.randint(1, 2)
            normal_mask_sum = normal_mask[step * j:step * (j+1),
                                        step * i:step * (i+1)].sum()
            normal_mask_max = step * step * 255
            ratio = normal_mask_sum/normal_mask_max
            
            # choose patch randomly (50%)
            if ratio > NORMAL_RATIO and random_num % 2 == 0:
                normal_patch_name = normal_patch_path + str(i) + '_' + str(j) + '_.png'
                patch = slide.read_region((304*i,304*j), 0, (304,304))
                patch.save(normal_patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (255,255,0), 1)
                patch_cnt += 1
            
            # if patch number is more than 25,000, stop extracting
            if patch_cnt > 25000:
                print('\n===== make_normal_patch END(OVER) =====')
                cv2.imwrite(map_path, slide_map)
                return
            all_cnt += 1

            print('\rProcess: %.3f%%, All: %d, Patch: %d'
                    %((100.*all_cnt/total), all_cnt, patch_cnt), end="")
    
    # save map
    cv2.imwrite(map_path, slide_map)


if __name__ == "__main__":
    procs = []
    
    # multiprocessing
    for i in range(1, 16): 
        process = Process(target = make_normal_patch, args =(i, MASK_LEVEL,))
        procs.append(process)
        process.start()

    for p in procs:
        p.join()
