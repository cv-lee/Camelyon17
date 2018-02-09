import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import numpy as np

from dataset_demo import get_demo_dataset
from dataset_demo import get_dimension
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()

print('\n==>Preparing data..')
test_trans = transforms.Compose([transforms.ToTensor(),])
test_dataset = get_demo_dataset(test_trans)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, 
                                            shuffle=False, num_workers=40)
print('\nData loading END')

checkpoint = torch.load('./checkpoint/ckpt.t7') ##확인
net = checkpoint['net']
threshold = checkpoint['threshold']



def test():
    '''

    '''

    net.eval()
    tumor_location = []
    for batch_idx, (inputs, location) in enumerate(test_loader):
        if USE_CUDA:
            inputs = inputs.cuda()

        batch_size = inputs.shape[0] 
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs = torch.squeeze(outputs)
        outputs += Variable((torch.ones(batch_size) * (1-threshold)).cuda())
        outputs = torch.floor(outputs)
 
        for idx, output in enumerate(list(outputs.data)): # float임
            if output == 1:
                tumor_location += [list(location[idx])]

    tumor_location = np.array(tumor_location).astype(int)
    return tumor_location



def make_heatmap(tumor_location):
    '''
    
    '''

    height,width = get_dimension(0,152) #pixel size = 152
    heatmap = np.zeros([width,height])
    
    for h,w in tumor_location:
        heatmap[w*2, h*2] = 255
        heatmap[w*2+1, h*2] = 255
        heatmap[w*2, h*2+1] = 255
        heatmap[w*2+1, h*2+1] = 255

    cv2.imwrite(HEATMAP_PATH + 'heatmap1.png', heatmap)



if __name__ == "__main__":
    tumor_location = test()
    make_heatmap(tumor_location)
