import pandas as pd
from pandas import DataFrame

patch_name_format = 'slide_num' + 'patch_X_coordinate_num' + 'patch_Y_coordinate_num'


tissue_patch_list = []
tumor_patch_list = []

tissue = {}
tumor= {}

for tissue_patch in tissue_patch_list:

    tissue[tissue_patch] = 0


for tumor_patch in tumor_patch_list:
    
    tissue[tumor_patch] = 1
