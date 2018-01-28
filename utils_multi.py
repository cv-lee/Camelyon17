import openslide
import cv2
import numpy as np
from xml.etree.ElementTree import parse
from PIL import Image

from multiprocessing import Process

def make_tumor_mask(img_num):

    #print('\n--- make_tumor_mask start ---\nImage Number: '+str(img_num))
    img_path = 'slide/b_'+str(img_num)+'.tif'
    label_path = 'annotation/b_'+str(img_num)+'.xml'
    img = openslide.OpenSlide(img_path)
    xml = parse(label_path).getroot()

    coordinate_list = []
    coordinate = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coordinate.append([round((float(area.get('X'))/8)),
                                round((float(area.get('Y'))/8))])
        coordinate_list.append(coordinate)
        coordinate = []
    coordinate_list = np.array(coordinate_list)

    mask = np.zeros(img.level_dimensions[3][::-1])
    for area in coordinate_list:
        mask = cv2.drawContours(mask, np.asarray([area]), -1, 255, -1)
    cv2.imwrite(('patch/b'+str(img_num)+'_patch/mask.png'),mask)

    width,height = img.level_dimensions[0]
    width = width//304
    height = height//304
    total = width*height
    all_cnt = 0
    patch_cnt = 0

    for i in range(width):
        for j in range(height):
            mask_sum = (np.array(mask[38*j:38*(j+1),
                        38*i:38*(i+1)]).sum())
            mask_max = 38*38*255
            if mask_sum/mask_max > 0.5:
                patch = img.read_region((304*i,304*j),0,(304,304))
                save_name = 'patch/b'+str(img_num)+'_patch/patch'+str(img_num)+'_'+ str(i) + '_' + str(j) + '.png'
                patch.save(save_name)
                patch_cnt += 1
            all_cnt += 1
            print('\rProcess: %.3f,  All: %d, Patch: %d' 
                %((100.*all_cnt/total),all_cnt,patch_cnt),end="")
    #print('\n--- make_tumor_mask_end ---')



if __name__ == "__main__":
    
    for slide_num in range(1,16):
        process = Process(target=make_tumor_mask, args = (slide_num,))
        process.start()
    
