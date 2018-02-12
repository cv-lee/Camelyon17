import openslide
import cv2
import sys
import time
import math
import numpy as np
import os
import random
import shutil
import csv

from sklearn.metrics import roc_auc_score
from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process
from user_define import config as cf
from user_define import hyperparameter as hp



# Parameters for progress_bar Init
TOTAL_BAR_LENGTH = 65.

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

last_time = time.time()
begin_time = last_time



def progress_bar(current, total, msg=None):
    ''' print current result of train, valid
    
    Args:
        current (int): current batch idx
        total (int): total number of batch idx
        msg(str): loss and acc
    '''

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()



def format_time(seconds):
    ''' calculate and formating time 

    Args:
        seconds (float): time
    '''

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def stats(outputs, targets):
    ''' Using outputs and targets list, calculate true positive,
        false positive, true negative, false negative, accuracy, 
        recall, specificity, precision, F1 Score, AUC, best Threshold.
        And return them

    Args:
        outputs (numpy array): net outputs list
        targets (numpy array): correct result list

    '''
    
    num = len(np.arange(0,1.005,0.005))

    correct = [0] * num
    tp = [0] * num
    tn = [0] * num
    fp = [0] * num
    fn = [0] * num
    recall = [0] * num
    specificity = [0] * num

    outputs_num = outputs.shape[0]
    for i, threshold in enumerate(np.arange(0, 1.005, 0.005)):
            
        threshold = np.ones(outputs_num) * (1-threshold)
        _outputs = outputs + threshold
        _outputs = np.floor(_outputs)

        tp[i] = (_outputs*targets).sum()
        tn[i] = np.where((_outputs+targets)==0, 1, 0).sum()
        fp[i] = np.floor(((_outputs-targets)*0.5 + 0.5)).sum()
        fn[i] = np.floor(((-_outputs+targets)*0.5 + 0.5)).sum()
        correct[i] += (tp[i] + tn[i])

    thres_cost = fp[0]+fn[0]
    thres_idx = 0

    for i in range(num):
        recall[i] = tp[i] / (tp[i]+fn[i])
        specificity[i] = tn[i] / (fp[i]+tn[i])
        if thres_cost > (fp[i]+fn[i]):
            thres_cost = fp[i]+fn[i]
            thres_idx = i

    correct = correct[thres_idx]
    tp = tp[thres_idx]
    tn = tn[thres_idx]
    fp = fp[thres_idx]
    fn = fn[thres_idx]
    recall = (tp+1e-7)/(tp+fn+1e-7)
    precision = (tp+1e-7)/(tp+fp+1e-7)
    specificity = (tn+1e-7)/(fp+tn+1e-7)
    f1_score = 2.*precision*recall/(precision+recall+1e-7)
    auc = roc_auc_score(targets, outputs) 
    threshold = thres_idx * 0.005

    return correct, tp, tn, fp, fn, recall, precision, specificity, f1_score,auc,threshold



def make_dir(slide_num, flags):
    ''' make directory of files using flags
        if flags is tumor_patch or normal patch
        additional directory handling is needed

    Args:
        slide_num (int): number of slide used
        flags (str): various flags are existed below
    '''

    if flags == 'slide':
        return cf.slide_path + 'b_' + str(slide_num) + '.tif'

    elif flags == 'xml':
        return cf.xml_path + 'b_' + str(slide_num) +'.xml'

    elif flags == 'mask':
        return cf.mask_path

    elif flags == 'map':
        return cf.mask_path + 'b' + str(slide_num) + '_map.png'

    elif flags == 'tumor_mask':
        return cf.mask_path + 'b' + str(slide_num) + '_tumor_mask.png'

    elif flags == 'tumor_patch':
        return cf.patch_path + 'b' + str(slide_num) + '/tumor/'

    elif flags == 'normal_mask':
        return cf.mask_path + 'b' + str(slide_num) + '_normal_mask.png'

    elif flags == 'normal_patch':
        return cf.patch_path + 'b' + str(slide_num) + '/normal/'
    
    elif flags == 'tissue_mask':
        return cf.mask_path + 'b' + str(slide_num) + '_tissue_mask.png'

    else:
        print('make_dir flags error')
        return



def chk_file(filedir,filename):    
    ''' check whether file(filename) is existed in filedir or not
        if existed, return True. else, return False

    Args: 
        fliedir (str): directory of the file
        filename (str): name of the file
    '''

    exist = False

    os.chdir(filedir)
    cwd = os.getcwd()
    
    for file_name in os.listdir(cwd):
        if file_name == filename:
            exist = True
    
    os.chdir(cf.origin_path)

    return exist



def read_xml(slide_num, mask_level):
    ''' read xml files which has tumor coordinates list
        return coordinates of tumor areas

    Args:
        slide_num (int): number of slide used
        maks_level (int): level of mask
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
    ''' Extract normal, tumor patches using normal, tumor mask

    Args:
        slide_num (int): number of slide used
        mask_level (int): level of mask
    '''

    slide_path = make_dir(slide_num,'slide')
    map_path = make_dir(slide_num, 'map')
    mask_folder_path = make_dir(slide_num, 'mask')
    tumor_mask_path = make_dir(slide_num,'tumor_mask')
    tumor_patch_path = make_dir(slide_num,'tumor_patch')
    normal_mask_path = make_dir(slide_num, 'normal_mask')
    normal_patch_path = make_dir(slide_num,'normal_patch')
   
    tumor_threshold = hp.tumor_threshold
    tumor_sel_ratio = hp.tumor_sel_ratio
    tumor_sel_max = hp.tumor_sel_max
    normal_threshold = hp.normal_threshold
    normal_sel_ratio = hp.normal_sel_ratio
    normal_sel_max = hp.normal_sel_max

    tumor_mask_exist = chk_file(mask_folder_path, 'b'+str(slide_num)+'_tumor_mask.png')
    normal_mask_exist = chk_file(mask_folder_path, 'b'+str(slide_num)+'_normal_mask.png')
    if (tumor_mask_exist and normal_mask_exist) == False:
        print('tumor or normal mask does NOT EXIST')
        return

    slide = openslide.OpenSlide(slide_path)
    slide_map = cv2.imread(map_path,-1)
    tumor_mask = cv2.imread(tumor_mask_path, 0)
    normal_mask = cv2.imread(normal_mask_path, 0)

    p_size = hp.patch_size
    width, height = np.array(slide.level_dimensions[0])//p_size
    total = width * height
    all_cnt = 0
    t_cnt = 0
    n_cnt = 0
    t_over = False
    n_over = False
    step = int(p_size/(2**mask_level))

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
                patch = slide.read_region((p_size*i,p_size*j), 0, (p_size,p_size))
                patch.save(patch_name)
                cv2.rectangle(slide_map, (step*i,step*j), (step*(i+1),step*(j+1)), (0,0,255), 1)
                t_cnt += 1
                t_over = (t_cnt > tumor_sel_max)
            
            # extract normal patch
            elif (normal_area_ratio > normal_threshold) and (ran <= normal_sel_ratio) and (tumor_area_ratio == 0) and not n_over:
                patch_name = normal_patch_path + 'n_b' + str(slide_num) + '_' + str(i) + '_' + str(j) + '_.png'
                patch = slide.read_region((p_size*i,p_size*j), 0, (p_size,p_size))
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
    '''make tumor, normal, tissue mask using xml files and otsu threshold

    Args:
        slide_num (int): number of slide
        mask_level (int): level of mask
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
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[hp.map_level]))

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



def divide_patch(slide_num, flags):
    ''' divide patches to train set, validation set, test set.
        specific slides are used only for trainset.
        others are used only for validationset and testset.

    Args:
        slide_num (int): number of slide used
        flags (str): determine wheter the slide might go for train or others
    '''

    tumor_patch_path = make_dir(slide_num, 'tumor_patch')
    normal_patch_path = make_dir(slide_num, 'normal_patch')

    tumor_files = os.listdir(tumor_patch_path)
    tumor_num = len(tumor_files)
    random.shuffle(tumor_files)

    normal_files = os.listdir(normal_patch_path)
    normal_num = len(normal_files)
    random.shuffle(normal_files)

    os.chdir(tumor_patch_path)

    if flags == 'train':
        for i in range(tumor_num):
            shutil.move(tumor_files[i],cf.dataset_path+'train/')

        os.chdir(normal_patch_path)

        for i in range(normal_num):
            shutil.move(normal_files[i], cf.dataset_path+'train/')
         
    else:
        for i in range(tumor_num):
            if i%2 == 0:
                shutil.move(tumor_files[i], cf.dataset_path+'validation/')
            else:
                shutil.move(tumor_files[i], cf.dataset_path+'test/')
        
        os.chdir(normal_patch_path)

        for i in range(normal_num):
            if i%2 == 0:
                shutil.move(normal_files[i], cf.dataset_path+'validation/')
            else:
                shutil.move(normal_files[i], cf.dataset_path+'test/')
        
    os.chdir(cf.origin_path)



def make_label():
    ''' make label csv file using file name (ex. t_ ... Tumor / n_ ... Normal)
    
    '''
       
    # path init
    train_path = cf.dataset_path + 'train/label/train_label.csv'
    valid_path = cf.dataset_path + 'validation/label/valid_label.csv'
    test_path = cf.dataset_path + 'test/label/test_label.csv'
    mining_path = cf.dataset_path + 'mining/label/mining_label.csv'

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
    file_list = os.listdir(cf.dataset_path+'train')
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
    file_list = os.listdir(cf.dataset_path+'validation')
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
    file_list = os.listdir(cf.dataset_path+'test')
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
    file_list = os.listdir(cf.dataset_path+'mining')
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
    ''' copy files based on csv files which have hard patches
    
    '''

    for i in range(cf.mining_csv_path):
        mining_csv = open(cf.mining_csv_path+'wrong_data_epoch'+str(i)+'.csv',
                            'r', encoding='utf-8')
        reader = csv.reader(mining_csv)
        
        for img in reader:
            if str(img[0])[0] == 't':
                shutil.copy(cf.dataset_path+'train/'+str(img[0]),
                            cf.dataset_path+'mining/'+str(img[0]))



'''
# run
make_mask(SLIDE_NUM, MASK_LEVEL)
make_patch(SLIDE_NUM, MASK_LEVEL)
divide_patch(SLIDE_NUM,'train')
make_label()
mining()
'''



'''
# multiprocessing run
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
