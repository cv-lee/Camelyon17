import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import argparse
import csv
import shutil
import sys
sys.path.append("..")
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from models import *
from dataset.dataset_train import get_dataset
from utils import progress_bar
from utils import stats
from user_define import config as cf
from user_define import hyperparameter as hp



# Basic Parameters Init  
BEST_AUC = 0
THRESHOLD = 0.5
START_EPOCH = 0
LR_DECAY = 0
LR_CHANCE = 0

CUR_EPOCH = []
CUR_LOSS = []
CUR_VAL_ACC = []
CUR_TRA_ACC = []
CUR_LR = []

USE_CUDA = torch.cuda.is_available()



# Parser Init
parser = argparse.ArgumentParser(description='Camelyon17 Training' )
parser.add_argument('--lr', default=hp.default_lr, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()



# Data loading
print('==> Preparing data..')
trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
])

trans_test = transforms.Compose([
    transforms.ToTensor(),
])

if hp.mining == True:
    trainset, valset, subtestset, testset, miningset = get_dataset(trans_train, trans_test, hp.train_num, hp.val_num, hp.subtest_num, hp.train_ratio, hp.mining)
    miningloader = torch.utils.data.DataLoader(miningset, batch_size=hp.batch_size, 
                                            shuffle=True, num_workers=hp.num_workers)
else: 
    trainset, valset, subtestset, testset = get_dataset(trans_train, trans_test, hp.train_num, hp.val_num, hp.subtest_num)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=hp.batch_size,
                                            shuffle=True, num_workers=hp.num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=hp.batch_size,
                                            shuffle=False, num_workers=hp.num_workers)
subtestloader = torch.utils.data.DataLoader(subtestset, batch_size=hp.batch_size,
                                            shuffle=False, num_workers=hp.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=hp.batch_size,
                                            shuffle=False, num_workers=hp.num_workers)
print('Data loading END')



# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/chance/ckpt.t7')
    net = checkpoint['net']
    BEST_AUC = checkpoint['auc']
    START_EPOCH = checkpoint['epoch']
    THRESHOLD = checkpoint['threshold']
    if checkpoint['lr'] < 1e-5:
        args.lr = 1e-5
    else:
        args.lr = checkpoint['lr']
else:
    print('==> Building model..')
    #net = resnet18()
    #net = resnet34()
    #net = resnet50()
    #net = resnet101()
    #net = resnet152()
    net = densenet121()
    #net = densenet161()
    #net = densenet201()

if USE_CUDA:
    if args.resume == False:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
        cudnn.benchmark = True



# Optimization, Loss Function Init
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)



def train(epoch, wrong_save=False):
    ''' trian net using patches of slide. 
        save csv file that has patch file name predicted incorrectly.
    
    Args:
        epoch (int): current epoch 
        wrong_save (bool):  If True, save the csv file that has patch file name
                            predicted incorrectly
    '''

    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    wrong_list = []

    for batch_idx, (inputs, targets, filename) in enumerate(trainloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()        
        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        total += targets.size(0)
        batch_size = targets.shape[0]

        outputs += Variable((torch.ones(batch_size) * (THRESHOLD)).cuda())
        outputs = torch.floor(outputs)
        correct += outputs.data.eq(targets.data).cpu().sum()
        filename_list = filename
        
        if wrong_save == True:
            for idx in range(len(filename_list)):
                if outputs.data[idx] != targets.data[idx]:
                    wrong_name = filename_list[idx]
                    wrong_list.append(wrong_name)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if wrong_save == True:
        wrong_csv = open(cf.wrong_path+'wrong_data_epoch'+str(epoch)+'.csv','w',encoding='utf-8')
        wr = csv.writer(wrong_csv)
        for name in wrong_list:
            wr.writerow([name])
        wrong_csv.close()
    
    CUR_TRA_ACC.append(100.*correct/total)



def valid(epoch):
    ''' valid net using patches of slide.
        Save checkpoint if AUC score is higher than saved checkpoint's.
    
    Args: 
        epoch (int): current epoch
    '''

    global BEST_AUC
    global THRESHOLD
    global LR_CHANCE
    global CK_CHANCE
    global LR_DECAY
    
    net.eval()
    valid_loss = 0
    total = 0
    correct = 0

    outputs_list = np.array([])
    targets_list = np.array([])

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

        batch_size = targets.shape[0]
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        total += targets.size(0)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        valid_loss += loss.data[0]

        _outputs = np.array(outputs.data).astype(float)
        _targets = np.array(targets.data).astype(float)
        outputs_list = np.append(outputs_list, _outputs)
        targets_list = np.append(targets_list, _targets)
        
        outputs += Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
        outputs = torch.floor(outputs)
        correct += int(outputs.eq(targets).cpu().sum())

        progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    correct, tp, tn, fp, fn, recall, precision, specificity, f1_score, auc, threshold = stats(outputs_list, targets_list)
    acc = correct/total
    THRESHOLD = threshold

    print('Acc: %.3f, Recall: %.3f, Prec: %.3f, Spec: %.3f, F1: %.3f, Thres: %.3f, AUC: %.3f' 
        %(acc, recall, precision, specificity, f1_score, threshold, auc))
    print('%17s %12s\n%-11s %-8d    %-8d\n%-11s %-8d    %-8d' 
        %('Tumor', 'Normal','pos',tp,fp,'neg',fn,tn))
    print("lr: ",args.lr * (0.5 ** (LR_DECAY)), "lr chance:",LR_CHANCE)
    
    # plot data   
    CUR_EPOCH.append(epoch)
    CUR_VAL_ACC.append(acc)
    CUR_LOSS.append(valid_loss/(batch_idx+1))
    CUR_LR.append(args.lr * (0.5 ** (LR_DECAY)))
    
    # Save checkpoint.
    if auc > BEST_AUC:
        print('saving...')
        BEST_AUC = auc
        state = {
            'net': net if USE_CUDA else net,
            'acc': acc,
            'loss': valid_loss, 
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'auc': auc,
            'epoch': epoch,
            'lr': args.lr * (0.5**(LR_DECAY)),
            'threshold': threshold
        }
        torch.save(state, './checkpoint/ckpt.t7')



def subtest():
    ''' test net using patches of slide.
        compare outputs of net and targets and print result.

    '''
    
    os.path.isdir('checkpoint')
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    threshold = checkpoint['threshold']
    net.eval()
    outputs_list =np.array([])
    targets_list =np.array([])
    test_loss = 0
    total = 0
    correct= 0

    for batch_idx, (inputs,targets) in enumerate(subtestloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()
        
        batch_size = targets.shape[0]
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        total += targets.size(0)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]

        _outputs = np.array(outputs.data).astype(float)
        _targets = np.array(targets.data).astype(float)
        outputs_list = np.append(outputs_list, _outputs)
        targets_list = np.append(targets_list, _targets)
        
        outputs += Variable((torch.ones(batch_size) * (1-threshold)).cuda())
        outputs = torch.floor(outputs)
        correct += int(outputs.eq(targets).cpu().sum())
        
        progress_bar(batch_idx, len(subtestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    correct, tp, tn, fp, fn, recall, precision, specificity, f1_score, auc,threshold = stats(outputs_list, targets_list)
    acc = correct/total
    print('Acc: %.3f, Recall: %.3f, Prec: %.3f, Spec: %.3f, F1: %.3f, Thres: %.3f, AUC: %.3f' 
        %(acc, recall, precision, specificity, f1_score, threshold, auc))
    print('%17s %12s\n%-11s %-8d    %-8d\n%-11s %-8d    %-8d' 
        %('Tumor', 'Normal','pos',tp,fp,'neg',fn,tn))
    print("lr: ",args.lr * (0.5 ** (LR_DECAY)), " chance:",LR_CHANCE)
    


def test():
    ''' test net using patches of slide.
        As there are no targets, save only patch outputs with csv file.

    '''
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    threshold = checkpoint['threshold']
    net.eval()
    total = 0
        
    outputs_list = []

    for batch_idx, inputs in enumerate(testloader): 
        if USE_CUDA:
            inputs = inputs.cuda()

        batch_size = inputs.shape[0]
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)

        outputs = torch.squeeze(outputs) 
        outputs += Variable((torch.ones(batch_size) * (1-threshold)).cuda())
        outputs = torch.floor(outputs)
            
        outputs_list = outputs_list + list(outputs.data)
            
    img_list = []
    all_list = {}
    result = open('/home/interns/camelyon17/result/test_result.csv','w',encoding='utf-8')
    result_writer = csv.writer(result)
     
    for img in os.listdir('/home/interns/camelyon17/dataset/dataset/test0201'):
        img_list.append(img)
        
    for i in range(len(outputs_list)):
        all_list[img_list[i]] = int(outputs_list[i])
        
    for key,val in all_list.items():
        result_writer.writerow([key,val])

    result.close()



def mining(epoch): 
    ''' train net using difficult dataset(Hard Mining) and original train dataset. 
        ratio between them can be adjusted.
    
    Args:
        epoch (int): current epoch
    '''
    
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(miningloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()        
        
        batch_size = targets.shape[0]
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        outputs += Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
        outputs = torch.floor(outputs)
      
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        total += targets.size(0)
        correct += outputs.data.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(miningloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def adjust_learning_rate(optimizer, epoch):
    ''' as learning rate chance run out, learning rate decay.
        learing rate decreases 1/2 of previous learning rate.
    
    Args:
        optimizier (torch.optim): optimizer that is used currently
        epoch (int): current epoch
    '''

    global LR_CHANCE
    global LR_DECAY

    if LR_CHANCE <= 0:
        LR_DECAY += 1
        LR_CHANCE = 3

    lr = args.lr * (0.5 ** (LR_DECAY))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def draw_graph():
    ''' draw net's results (acc, loss, learning rate)
    
    '''

    if len(CUR_TRA_ACC) != 0:
        plt.figure()
        plt.plot(CUR_EPOCH,CUR_TRA_ACC)
        plt.title('Camelyon17 DenseNet/Train acc')
        plt.xlabel('epoch')
        plt.ylabel('train acc')
        plt.savefig(cf.result_path + 'train_acc.png')
        plt.clf()

    plt.figure()
    plt.plot(CUR_EPOCH, CUR_VAL_ACC)
    plt.title('Camelyon17 DenseNet/Val acc')
    plt.xlabel('epoch')
    plt.ylabel('valid acc')
    plt.savefig(cf.result_path + 'val_acc.png')
    plt.clf()
    
    plt.figure()
    plt.plot(CUR_EPOCH, CUR_LOSS)
    plt.title('Camelyon17 DenseNet/Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(cf.result_path + 'loss.png')
    plt.clf()
    
    plt.figure()
    plt.plot(CUR_EPOCH, CUR_LR)
    plt.title('Camelyon17 DenseNet/lr')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.savefig(cf.result_path + 'lr.png')
    plt.clf()



# run
if __name__ == "__main__":
    for epoch in range(START_EPOCH, START_EPOCH+hp.epoch):
        adjust_learning_rate(optimizer,epoch)
        train(epoch, hp.wrong_save)
        #mining(epoch)
        valid(epoch)
    subtest()
    #test()
    draw_graph()
