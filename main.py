import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import argparse
import csv
#import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from models import *
from dataset.dataset import get_dataset
from utils import progress_bar

RESULT_PATH = '/home/interns/camelyon17/result/'
WRONG_PATH = '/home/interns/camelyon17/hard_dataset/'

THRESHOLD = 0.15
EPOCH = 13
CHANCE = 3	# if chance == 0, learning rate decay 
LR_DECAY = 0
BEST_ACC = 0
BEST_LOSS = 100 
BEST_RECALL = 0
BEST_SPECIFICITY = 0
BEST_PRECISION = 0
BEST_F1_SCORE = 0
BEST_AUC = 0
START_EPOCH = 0	# start from epoch 0 or last checkpoint epoch
MINING = False

CUR_EPOCH = []
CUR_LOSS = []
CUR_VAL_ACC = []
CUR_TRA_ACC = []
CUR_LR = []
USE_CUDA = torch.cuda.is_available()


# parser init
parser = argparse.ArgumentParser(description='Pytorch Camelyon17 Training' )
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
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

if MINING == True:
    trainset, valset, subtestset, testset,miningset = get_dataset(trans_train, trans_test,MINING)
    miningloader = torch.utils.data.DataLoader(miningset, batch_size=250, 
                                            shuffle=True, num_workers=40)
else: 
    trainset, valset, subtestset, testset = get_dataset(trans_train, trans_test, MINING)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=250,
                                            shuffle=True, num_workers=40)
valloader = torch.utils.data.DataLoader(valset, batch_size=250,
                                            shuffle=False, num_workers=40)
subtestloader = torch.utils.data.DataLoader(subtestset, batch_size=250,
                                            shuffle=False, num_workers=40)
testloader = torch.utils.data.DataLoader(testset, batch_size=250,
                                            shuffle=False, num_workers=40)
print('Data loading END')


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    BEST_ACC = checkpoint['acc']
    BEST_LOSS = checkpoint['loss'] # DenseNet 전용
    BEST_RECALL = checkpoint['recall']
    BEST_SPECIFICITY = checkpoint['specificity']
    BEST_PRECISION = checkpoint['precision']
    BEST_F1_SCORE = checkpoint['f1_score']
    BEST_AUC = checkpoint['auc']
    START_EPOCH = checkpoint['epoch']
    if checkpoint['lr'] < 1e-7:
        args.lr = 1e-7
    else:
        args.lr = checkpoint['lr']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    #net = resnet34()
    # net = PreActResNet18()
    # net = GoogLeNet()
    net = densenet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()용

if USE_CUDA:
    if args.resume == False:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
        cudnn.benchmark = True


# Optimization Init
criterion = nn.BCELoss()
optimizer_sgd = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_adam = optim.Adam(net.parameters(), lr=args.lr)
optimizer_rmsprop = optim.RMSprop(net.parameters(), lr=args.lr)
optimizer=optimizer_sgd


# Train
def train(epoch, csv_save=False):
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
        filename_list = filename    

        batch_size = targets.shape[0]
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)
        thresholding = Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
        outputs = outputs + thresholding 
        outputs = torch.floor(outputs)
        
        if csv_save == True:
            for idx in range(len(filename_list)):
                if outputs.data[idx] != targets.data[idx]:
                    wrong_name = filename_list[idx]
                    wrong_list.append(wrong_name)

        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        total += targets.size(0)
        correct += outputs.data.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if csv_save == True:
        wrong_csv = open(WRONG_PATH+'wrong_data_epoch'+str(epoch)+'.csv','w',encoding='utf-8')
        wr = csv.writer(wrong_csv)
        for name in wrong_list:
            wr.writerow([name])
        wrong_csv.close()
    
    CUR_TRA_ACC.append((100.*correct/total))


# validation
def valid(epoch):
    global BEST_ACC
    global BEST_LOSS 
    global BEST_RECALL
    global BEST_PRECISION
    global BEST_SPECIFICITY
    global BEST_F1_SCORE
    global BEST_AUC
    global CHANCE
    global LR_DECAY
    
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    outputs_list = []
    targets_list = []

    for batch_idx, (inputs, targets) in enumerate(valloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()

        batch_size = targets.shape[0]
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        outputs =torch.squeeze(outputs)
        outputs_list += list(outputs.data)
        targets_list += list(targets.data)
        loss = criterion(outputs, targets)
        
        thresholding = Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda()) 
        outputs = outputs + thresholding 
        outputs = torch.floor(outputs)
        
        for idx in range(len(outputs.data)):
            if outputs.data[idx] == 1:
                if outputs.data[idx] == targets.data[idx]:
                    tp += 1
                else:
                    fp += 1
            else:
                if outputs.data[idx] == targets.data[idx]:
                    tn += 1
                else:
                    fn += 1

        valid_loss += loss.data[0]
        total += targets.size(0)
        correct += outputs.data.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    recall = (tp+1e-7)/(tp+fn+1e-7)
    precision = (tp+1e-7)/(tp+fp+1e-7)
    specificity = (tn+1e-7)/(fp+tn+1e-7)
    f1_score = 2.*precision*recall/(precision+recall+1e-7)
    auc = roc_auc_score(targets_list, outputs_list) 

    print('Acc: %.3f, Recall: %.3f, Prec: %.3f, Spec: %.3f, F1: %.3f, AUC: %.3f' %(acc, recall, precision, specificity,f1_score,auc))
    print('%17s %12s\n%-11s %-8d    %-8d\n%-11s %-8d    %-8d' %('Tumor', 'Normal','pos',tp,fp,'neg',fn,tn))
    print("lr: ",args.lr * (0.5 ** (LR_DECAY)), " chance:",CHANCE)
    
    
    # plot data   
    CUR_EPOCH.append(epoch)
    CUR_VAL_ACC.append(100.*correct/total)
    CUR_LOSS.append(valid_loss/(batch_idx+1))
    CUR_LR.append(args.lr * (0.5 ** (LR_DECAY)))
    
    # Save checkpoint.

    if auc > BEST_AUC:
        print('Saving..')
        state = {
            'net': net if USE_CUDA else net,
            'acc': acc,
            'loss': valid_loss, # DenseNet 전용
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'auc': auc,
            'epoch': epoch,
            'lr': args.lr * (0.5**(LR_DECAY))
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7') #chkpt.pth.tar 로 수정해야함
        BEST_ACC = acc
        BEST_LOSS = valid_loss
        BEST_RECALL = recall
        BEST_SPECIFICITY = specificity
        BEST_PRECISION = precision
        BEST_F1_SCORE = f1_score
        BEST_AUC = auc
    
    else:
        CHANCE -= 1


# test
def test(flags):
    if flags == 'subtest':
        os.path.isdir('checkpoint')
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        tp, tn, fp, fn = 0,0,0,0
        acc, recall, precision, specificity = 0, 0, 0, 0

        for batch_idx, (inputs,targets) in enumerate(subtestloader):
            if USE_CUDA:
                inputs = inputs.cuda()
                targets = torch.FloatTensor(np.array(targets).astype(float)).cuda()
        
            batch_size = targets.shape[0]
            inputs, targets =Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)

            outputs = torch.squeeze(outputs) 
            loss = criterion(outputs, targets)
            thresholding = Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
            outputs = outputs + thresholding 
            outputs = torch.floor(outputs)
            
            for idx in range(len(outputs.data)):
                if outputs.data[idx] == 1:
                    if outputs.data[idx] == targets.data[idx]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if outputs.data[idx] == targets.data[idx]:
                        tn += 1
                    else:
                        fn += 1

            test_loss += loss.data[0]
            total += targets.size(0)
            correct += outputs.data.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        recall = (tp+1e-7)/(tp+fn+1e-7)
        precision = (tp+1e-7)/(tp+fp+1e-7)
        specificity = (tn+1e-7)/(fp+tn+1e-7)
        f1_score = 2.*precision*specificity/(precision+specificity+1e-7)
        
        print("\n\n[Test]")
        progress_bar(batch_idx, len(subtestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('\nAcc: %.3f, Recall: %.3f, Prec: %.3f, Spec: %.3f, F1: %.3f' %(acc, recall, precision, specificity, f1_score))
        print('\n%17s %12s\n%-11s %-8d    %-8d\n%-11s %-8d    %-8d' %('Tumor', 'Normal','pos',tp,fp,'neg',fn,tn))

    else:
        os.path.isdir('checkpoint')
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net = checkpoint['net']
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
            thresholding = Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
            outputs = outputs + thresholding 
            outputs = torch.floor(outputs)
            
            outputs_list = outputs_list + list(outputs.data)
            
        img_list = []
        for img in os.listdir('/home/interns/test/test0201'):
            img_list.append(img)
        
        result = open('/home/interns/test/test_result.csv','w',encoding='utf-8')
        result_writer = csv.writer(result)
        
        all_result = {}
        for i in range(len(outputs_list)):
            all_result[img_list[i]] = int(outputs_list[i])
        
        for key,val in all_result.items():
            result_writer.writerow([key,val])

        result.close()


# mining
def mining(epoch): 
    '''
        Use 100,000 dataset in train and
            10,000 dataset in hard_dataset
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
        thresholding = Variable((torch.ones(batch_size) * (1-THRESHOLD)).cuda())
        outputs = outputs + thresholding 
        outputs = torch.floor(outputs)
      
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        total += targets.size(0)
        correct += outputs.data.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(miningloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# change learning rate and optimizer
def adjust_learning_rate(optimizer, epoch):
    global CHANCE
    global LR_DECAY

    if CHANCE <= 0:
        LR_DECAY += 1
        CHANCE = 3

    lr = args.lr * (0.5 ** (LR_DECAY))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# draw acc/lr/loss graph
def draw_graph():
    if len(CUR_TRA_ACC) != 0:
        plt.figure()
        plt.plot(CUR_EPOCH,CUR_TRA_ACC)
        plt.title('Camelyon17 ResNet/Train acc')
        plt.xlabel('epoch')
        plt.ylabel('train acc')
        plt.savefig(RESULT_PATH + 'train_acc.png')
        plt.clf()

    plt.figure()
    plt.plot(CUR_EPOCH, CUR_VAL_ACC)
    plt.title('Camelyon17 ResNet/Val acc')
    plt.xlabel('epoch')
    plt.ylabel('valid acc')
    plt.savefig(RESULT_PATH + 'val_acc.png')
    plt.clf()
    
    plt.figure()
    plt.plot(CUR_EPOCH, CUR_LOSS)
    plt.title('Camelyon17 ResNet/Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(RESULT_PATH + 'loss.png')
    plt.clf()
    
    plt.figure()
    plt.plot(CUR_EPOCH, CUR_LR)
    plt.title('Camelyon17 ResNet/lr')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.savefig(RESULT_PATH + 'lr.png')
    plt.clf()


# run
for epoch in range(START_EPOCH, START_EPOCH+EPOCH):
    adjust_learning_rate(optimizer,epoch)
    train(epoch,True)
#   mining(epoch)
    valid(epoch)
test('subtest')
draw_graph()
