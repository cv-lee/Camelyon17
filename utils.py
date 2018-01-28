import openslide
import cv2
import numpy as np
from xml.etree.ElementTree import parse
from PIL import Image


XML_PATH = 'annotation/' #xml files for tumor label 
SLIDE_PATH = 'slide/' #whole slide files (b_1 ~ b_15)i
PATCH_PATH = 'patch_test/' #Test시, patch_test 폴더에 저장할것

#이 전역변수 부부분은 main.py로 옮겨야함
TUMOR_MASK_LEVEL = 4
SLIDE_NUM = 3



def read_xml(slide_num, tumor_mask_level):
    path  = XML_PATH + 'b_' + str(slide_num) + 'xml'
    
    xml = parse(path).getroot()
    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X'))/(2**tumor_mask_level)),
                            round(float(area.get('Y'))/(2**tumor_mask_level))])
        coors_list.append(coors)
        coors=[]
    return np.array(coors_list)


def make_tumor_patch(slide_num,tumor_mask_level):
    slide_path = SLIDE_PATH + 'b_' + str(slide_num)+'.tif'
    mask_save_path = PATCH_PATH + 'b' + str(slide_num) + '/mask/b' + str(slide_num) + '_tumor_mask.png'
    patch_save_path = PATCH_PATH + 'b' + str(slide_num) + '/tumor/b' + str(slide_num)
    
    slide = openslide.OpenSlide(slide_path)
    coors_list = read_xml(slide_num, tumor_mask_level)
    tumor_mask = np.zeros(slide.level_dimensions[tumor_mask_level][::-1])
    
    for coors in coors_list:
        mask = np.array(cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1))
        #cv2.drawContours return 값이 np.array인지 확인해봐야함 / 맞으면 앞에 해줄필요없음
    cv2.imwrite(mask_save_path, tumor_mask)

    width,height = slide.level_dimensions[0]
    width = width//304  
    height = height//304
    total = width*height
    all_cnt = 0
    patch_cnt = 0

    for i in range(width):
        for j in range(height):
            tumor_mask_sum = (np.array(tumor_mask[38*j:38*(j+1),
                        38*i:38*(i+1)]).sum())
            tumor_mask_max = 38*38*255
            ratio = tumor_mask_sum/tumor_mask_max
            
            if ratio > 0.5:
                patch = slide.read_region((304*i,304*j),0,(304,304))
                save_name = 'patch/b'+str(img_num)+'_patch/patch' + str(i) + '_' + str(j) + '.png'
                patch.save(save_name)
                patch_cnt += 1
            all_cnt += 1
            print('\rProcess: %.3f,  All: %d, Patch: %d' 
                %((100.*all_cnt/total),all_cnt,patch_cnt),end="")
    print('\n--- make_tumor_mask END ---\n')

#Object Edge Dectection Method(OTSU)
img = openslide.OpenSlide('slide/b_3.tif')
img_lv3 = img.read_region((0,0),3,img.level_dimensions[3])
img_lv3 = cv2.cvtColor(np.array(img_lv3),cv2.COLOR_BGR2GRAY)

cv2.imwrite('img_gray.png',np.array(img_lv3))

blur = cv2.GaussianBlur(img_lv3,(5,5),0)
ret, th = cv2.threshold(img_lv3,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel_open = np.ones((2,2),dtype = np.uint8)
kernel_close = np.ones((4,4),dtype=np.uint8)

img_lv3 = cv2.morphologyEx(img_lv3,cv2.MORPH_CLOSE,kernel_close)
img_lv3 = cv2.morphologyEx(img_lv3,cv2.MORPH_OPEN,kernel_open)

_,contours, hierachy = cv2.findContours(img_lv3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
normal_mask = np.zeros(img_lv3.shape[:2])
normal_mask = normal_mask.astype(np.uint8)
cv2.drawContours(normal_mask, contours, -1, 255, -1)

print(normal_mask.shape)
tumor_mask = cv2.imread('patch/b3_patch/mask.png',cv2.IMREAD_GRAYSCALE)
print(tumor_mask.shape)
normal_mask -= tumor_mask
cv2.imwrite('img_otsu.png',normal_mask)


#width,height = img.level_dimensions[0]
#width = width//304
#height = height//304
#total = width*height
#all_cnt = 0
#patch_cnt = 0

#for i in range(width):
#    for j in range(height):
#        mask_sum = (np.array(mask[38*j:38*(j+1),38*i:38*(i+1)]).sum())
#        mask_max = 38*38*255
#        if mask_sum/mask_max == 1:
#            patch = img.read_region((304*i,304*j),0,(304,304))
#            save_name = 'patch/b'+str(img_num)+'_patch/patch' + str(i) + '_' + str(j) + '.png'
#            patch.save(save_name)
#            patch_cnt += 1
#        all_cnt += 1
#        print('\rProcess: %.3f,  All: %d, Patch: %d' %((100.*all_cnt/total),all_cnt,patch_cnt),end="")
#print('\n--- make_tumor_mask END ---\n')

#make tumor mask
#for i in range(15):
#    make_tumor_mask(i+1)

