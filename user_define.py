'''
    Save configurations and hyperparameters.
    Other files import these classes for configs, hyperparameters
'''

class config():
    ''' Files Path Collections Class

    '''
    
    result_path = '/home/interns/camelyon17/result/'
    wrong_path = '/home/interns/camelyon17/dataset/pre_dataset/difficult/'
    heatmap_path = '/home/interns/test/heatmap/DenseNet121_2/'

    origin_path  = '/home/interns/camelyon17'
    xml_path = '/mnt/disk3/interns/camelyon17/pre_dataset/annotation/'
    slide_path = '/mnt/disk3/interns/camelyon17/pre_dataset/slide/'
    mask_path = '/home/interns/dataset/pre_dataset/mask/'
    patch_path = '/mnt/disk3/interns/camelyon17/pre_dataset/patch/'
    dataset_path = '/mnt/disk3/interns/camelyon17/pre_dataset/dataset/'
    mining_csv_path = '/home/interns/camelyon17/dataset/pre_dataset/difficult/'

    test_path = '/home/interns/camelyon17/dataset/dataset/test0201/'

    #dataset_eval.py
    demo_slide_path = '/mnt/disk3/interns/camelyon17/dataset/t_6.tif'



class hyperparameter():
    ''' Hyperparameters Collections Class

    '''

    #utils.py
    slide_num = 3

    patch_size = 304    #fixed
    mask_level = 4      # fixed
    map_level = 4       # fixed

    normal_threshold = 0.1  # normal mask inclusion ratio that select normal patches
    normal_sel_ratio = 1    # nomral patch selection ratio 
    normal_sel_max = 100000 # number limit of normal patches 

    tumor_threshold = 0.8   # tumor mask inclusion ratio that select tumor patches
    tumor_sel_ratio = 1     # tumor patch selection ratio
    tumor_sel_max = 100000  # number limit of tumor patches

    mining_csv_num = 70     # number of csv files for hard mining


    #dataset_eval
    tissue_threshold = 0.4  # tisse mask inclusion ratio that select tissue patches

    #train.py
    train_num = 186     # max: 186,800
    val_num = 629       # max: 62,940
    subtest_num = 629   # max: 62,940    
    train_ratio = 1     # for mining, train set ratio compared with hard mining set

    default_lr = 0.005  # defalut learning ratio
    momentum = 0.9      # SGD optimizer parameter, 'momentum'
    weight_decay = 5e-4 # SGD optimizer parameter, 'weight_decay'
    
    epoch = 2           # train epoch
    batch_size = 250    # batch size (with using 8 Titan X GPU, 250 is limitation) 
    num_workers = 40    # number of CPU 
    mining = False      # train using hard mining set (on/off)
    wrong_save = False  # collect hard mining dataset (on/off)
