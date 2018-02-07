import openslide
import cv2
import numpy as np
import os
import random
import shutil
import csv

from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process

XML_PATH = '/mnt/disk3/interns/annotation/'
SLIDE_PATH = '/mnt/disk3/interns/slide/'
MASK_PATH = '/home/interns/mask/'
PATCH_PATH = '/mnt/disk3/interns/patch/'
DATASET_PATH = '/mnt/disk3/interns/dataset/'
MINING_CSV_PATH = '/home/interns/camelyon17/hard_dataset/'

#GLOBAL VARIABLES should move to 'main.py'
MASK_LEVEL = 4 # fixed 
MAP_LEVEL = 4 # fixed
SLIDE_NUM = 3

NORMAL_THRESHOLD = 0.1
NORMAL_SEL_RATIO = 1
NORMAL_SEL_MAX = 100000

TUMOR_THRESHOLD = 0.8
TUMOR_SEL_RATIO = 1
TUMOR_SEL_MAX = 100000

MINING_CSV_NUM = 70

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
        return MASK_PATH 

    elif flags == 'map':
        return MASK_PATH + 'b' + str(slide_num) + '_map.png'

    elif flags == 'tumor_mask':
        return MASK_PATH + 'b' + str(slide_num) + '_tumor_mask.png'

    elif flags == 'tumor_patch':
        return PATCH_PATH + 'b' + str(slide_num) + '/tumor/'

    elif flags == 'normal_mask':
        return MASK_PATH + 'b' + str(slide_num) + '_normal_mask.png'

    elif flags == 'normal_patch':
        return PATCH_PATH + 'b' + str(slide_num) + '/normal/'
    
    elif flags == 'tissue_mask':
        return MASK_PATH + 'b' + str(slide_num) + '_tissue_mask.png'

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


def make_patch(slide_num, mask_level):
    '''

    '''
    slide_path = make_dir(slide_num,'slide')
    map_path = make_dir(slide_num, 'map')
    mask_folder_path = make_dir(slide_num, 'mask')
    tumor_mask_path = make_dir(slide_num,'tumor_mask')
    tumor_patch_path = make_dir(slide_num,'tumor_patch')
    normal_mask_path = make_dir(slide_num, 'normal_mask')
    normal_patch_path = make_dir(slide_num,'normal_patch')
   
    tumor_threshold = TUMOR_THRESHOLD
    tumor_sel_ratio = TUMOR_SEL_RATIO
    tumor_sel_max = TUMOR_SEL_MAX
    normal_threshold = NORMAL_THRESHOLD
    normal_sel_ratio = NORMAL_SEL_RATIO
    normal_sel_max = NORMAL_SEL_MAX

    tumor_mask_exist = chk_file(mask_folder_path, 'b'+str(slide_num)+'_tumor_mask.png')
    normal_mask_exist = chk_file(mask_folder_path, 'b'+str(slide_num)+'_normal_mask.png')
    if (tumor_mask_exist and normal_mask_exist) == False:
        print('tumor or normal mask does NOT EXIST')
        return

    slide = openslide.OpenSlide(slide_path)
    slide_map = cv2.imread(map_path,-1)
    tumor_mask = cv2.imread(tumor_mask_path, 0)
    normal_mask = cv2.imread(normal_mask_path, 0)

    width, height = np.array(slide.level_dimensions[0])//304
    total = width * height
    all_cnt = 0
    t_cnt = 0
    n_cnt = 0
    t_over = False
    n_over = False
    step = int(304/(2**mask_level))

    for i in range(width):
        for j in range(height):
            ran = random.random()
            tumor_mask_sum = tumor_mask[step * j : step * (j+1),
                                            step * i : step * (i+1)].sum()
            normal_mask_sum = normal_mask[step * j : step * (j+1),
                                            step * i : step * (i+1)].sum()
            mask_max = step * step * 255
            tumor_area_ratio = tumor_mask_sum / mask_max
            normal_area_ratio = normal_mask_sum / mask_max

            # extract tumor patch
            if (tumor_area_ratio > tumor_threshold) and (ran <= tumor_sel_ratio) and not t_over:
                patch_name = tumor_patch_path + 't_b' + str(slide_num) + '_' + str(i) + '_' + str(j) + '_.png'
                patch = slide.read_region((304*i,304*j), 0, (304,304))
                patch.save(patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,0,255), 1)
                t_cnt += 1
                t_over = (t_cnt > tumor_sel_max)
            
            # extract normal patch
            elif (normal_area_ratio > normal_threshold) and (ran <= normal_sel_ratio) and (tumor_area_ratio == 0) and not n_over:
                patch_name = normal_patch_path + 'n_b' + str(slide_num) + '_' + str(i) + '_' + str(j) + '_.png'
                patch = slide.read_region((304*i,304*j), 0, (304,304))
                patch.save(patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (255,255,0), 1)
                n_cnt += 1
                n_over = (n_cnt > normal_sel_max)
            
            # nothing
            else:
                pass

            # check max boundary of patch
            if n_over and t_over:
                print('\nPatch Selection Boundary OVER')
                cv2.imwrite(map_path, slide_map)
                return
            
            all_cnt += 1
            print('\rProcess: %.3f%%,  All: %d, Normal: %d, Tumor: %d'
                %((100.*all_cnt/total), all_cnt, n_cnt, t_cnt), end="")

    cv2.imwrite(map_path, slide_map)


def make_mask(slide_num, mask_level):
    
    '''
        make tumor patch using tumor mask
    '''

    # path setting
    slide_path = make_dir(slide_num, 'slide')
    map_path = make_dir(slide_num, 'map')
    mask_folder_path = make_dir(slide_num, 'mask')
    
    tumor_mask_path = make_dir(slide_num, 'tumor_mask')
    tissue_mask_path = make_dir(slide_num, 'tissue_mask')
    normal_mask_path = make_dir(slide_num, 'normal_mask')

    #slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[MAP_LEVEL]))

    # xml loading
    coors_list = read_xml(slide_num, mask_level)
    
    # draw boundary of tumor in map
    for coors in coors_list:
        cv2.drawContours(slide_map, np.array([coors]), -1, 255, 1)
    cv2.imwrite(map_path, slide_map)

    # check tumor mask / draw tumor mask
    tumor_mask_exist = chk_file(mask_folder_path, 'b' + str(slide_num) + '_tumor_mask.png')
    if tumor_mask_exist == False:
        tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1])
        for coors in coors_list:
            cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)
            cv2.imwrite(tumor_mask_path, tumor_mask)

    # check tissue mask / draw tissue mask
    tissue_mask_exist = chk_file(mask_folder_path, 'b' + str(slide_num) + '_tissue_mask.png')
    if tissue_mask_exist == False:
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _,tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(tissue_mask_path, np.array(tissue_mask))
        
    # check normal mask / draw normal mask
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


def divide_patch(slide_num):
    '''

    '''
    tumor_patch_path = make_dir(slide_num, 'tumor_patch')
    normal_patch_path = make_dir(slide_num, 'normal_patch')

    tumor_files = os.listdir(tumor_patch_path)
    tumor_num = len(tumor_files)
    normal_files = os.listdir(normal_patch_path)
    normal_num = len(normal_files)

    # divide tumor patch files
    os.chdir(tumor_patch_path)
    random.shuffle(tumor_files)

    idx = 0
    rate = tumor_num//10

    for i in range(tumor_num):
        
        if idx < rate*8:
            shutil.move(tumor_files[idx], DATASET_PATH+'train/')
            idx += 1
        
        else:
            if (idx % 2 == 0):
                shutil.move(tumor_files[idx], DATASET_PATH+'validation/')
                idx += 1
            else:
                shutil.move(tumor_files[idx], DATASET_PATH+'test/')
                idx += 1

    os.chdir(normal_patch_path)
    random.shuffle(normal_files)

    idx = 0
    rate = normal_num//10
    
    for i in range(normal_num):
        
        if idx < rate*8:
            shutil.move(normal_files[idx], DATASET_PATH+'train/')
            idx += 1
        
        else:
            if (idx % 2 == 0):
                shutil.move(normal_files[idx], DATASET_PATH+'validation/')
                idx += 1
            else:
                shutil.move(normal_files[idx], DATASET_PATH+'test/')
                idx += 1

    os.chdir('/home/interns/camelyon17')


def make_label():
    '''

    '''
       
    # path init
    train_path = DATASET_PATH + 'train/label/train_label.csv'
    valid_path = DATASET_PATH + 'validation/label/valid_label.csv'
    test_path = DATASET_PATH + 'test/label/test_label.csv'
    mining_path = DATASET_PATH + 'mining/label/mining_label.csv'

    # csv files init
    train_csv = open(train_path, 'w', encoding='utf-8')
    valid_csv = open(valid_path, 'w', encoding='utf-8')
    test_csv = open(test_path, 'w', encoding='utf-8')
    mining_csv = open(mining_path, 'w', encoding='utf-8')
    
    # csv writer init
    train_writer = csv.writer(train_csv)
    valid_writer = csv.writer(valid_csv)
    test_writer = csv.writer(test_csv)
    mining_writer = csv.writer(mining_csv)
    
    
    # make train label.csv
    file_list = os.listdir(DATASET_PATH+'train')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in train folder')

    for key, val in label.items():
        train_writer.writerow([key, val])

    # make valid label.csv
    file_list = os.listdir(DATASET_PATH+'validation')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in validation folder')

    for key, val in label.items():
        valid_writer.writerow([key, val])

    # make test label.csv
    file_list = os.listdir(DATASET_PATH+'test')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1

        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0

        elif file_name == 'label':
            continue

        else:
            print('Error dataset in test folder')

    for key, val in label.items():
        test_writer.writerow([key, val])
    

    # make mining label.csv
    file_list = os.listdir(DATASET_PATH+'mining')
    label = {}

    for file_name in file_list:
        if file_name.split('_')[0] == 't':
            label[file_name] = 1
        
        elif file_name.split('_')[0] == 'n':
            label[file_name] = 0
        
        elif file_name == 'label':
            continue

        else:
            print('Error dataset in mining folder')

    for key, val in label.items():
        mining_writer.writerow([key,val])

    train_csv.close()
    valid_csv.close()
    test_csv.close()
    mining_csv.close()

def mining():
    '''
        copy files based on csv files which have hard patches
    '''

    for i in range(MINING_CSV_NUM):
        mining_csv = open(MINING_CSV_PATH+'wrong_data_epoch'+str(i)+'.csv',
                            'r', encoding='utf-8')
        reader = csv.reader(mining_csv)
        
        for img in reader:
            if str(img[0])[0] == 't':
                shutil.copy(DATASET_PATH+'train/'+str(img[0]),
                            DATASET_PATH+'mining/'+str(img[0]))
    

#make_mask(SLIDE_NUM, MASK_LEVEL)
#make_patch(SLIDE_NUM, MASK_LEVEL)
#divide_patch(SLIDE_NUM)
#make_label()
#mining()

'''
if __name__ == "__main__":
    procs = []
    
    # multiprocessing
    for i in range(1, 16): 
        process = Process(target = divide_patch, args =(i, ))
        procs.append(process)
        process.start()
    
    for proc in procs:
        proc.join()
'''
