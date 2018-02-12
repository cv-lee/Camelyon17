'''

'''
class config():
    '''

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

    #dataset_demo.py
    demo_slide_path = '/mnt/disk3/interns/camelyon17/dataset/t_6.tif'



class hyperparameter():
    '''

    '''

    #utils.py
    patch_size = 304
    mask_level = 4 # fixed
    map_level = 4 # fixed
    slide_num = 3

    normal_threshold = 0.1
    normal_sel_ratio = 1
    normal_sel_max = 100000

    tumor_sel_ratio = 1
    tumor_threshold = 0.8
    tumor_sel_max = 100000

    mining_csv_num = 70


    #dataset_demo
    tissue_threshold = 0.4

    #main.py
    train_num = 186     # max: 186,800
    val_num = 629       # max: 62,940
    subtest_num = 629   # max: 62,940    
    train_ratio = 1

    default_lr = 0.005
    momentum = 0.9
    weight_decay = 5e-4
    
    epoch = 2
    batch_size = 250
    num_workers = 40
    mining = False
    wrong_save = False

