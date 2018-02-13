# 1. Camelyon17 #


### 1-1) About
This project is the CNN model to diagnose breast cancer. It consists of a CNN model,
Desenet and ResNet. The criteria for performance is AUC(Area Under the Curve). 
An output of model is a float number from 0 to 1. (0: Normal, 1: Tumor)


### 1-2) Architecture
```
models/densenet.py
models/resnet.py
```

![](https://i.imgur.com/7yH9SKm.jpg)




# 2. Dataset


### 2-1) Overview
[CAMELYON17](https://camelyon17.grand-challenge.org/) is the second grand challenge in pathology organised 
by the Diagnostic Image Analysis Group (DIAG) and Department of Pathology of the Radboud University Medical 
Center (Radboudumc) in Nijmegen, The Netherlands.
The data in this challenge contains whole-slide images (WSI) of hematoxylin and eosin (H&E) stained lymph node sections.
All ground truth annotations were carefully prepared under supervision of expert pathologists. For the purpose of revising the slides, 
additional slides stained with cytokeratin immunohistochemistry were used. If however, you encounter problems 
with the dataset, then please report your findings at the forum.


```
utils.py
dataset_train.py
dataset_eval.py
```

### 2-2) Data Argumentation

Convert Images to Horizontal Flip
Convert Images to Vertical Flip
Convert Images to Gray Scale Randomly (percentage = 10%)
Convert Images brightness, contrast, saturation, hue slightly 


### 2-3) Mask
Using several masks, patch is extracted from them with mask inclusion ratio(hyperparameter) 

- Tissue Mask
![](https://i.imgur.com/y3hnyQA.png)
- Tumor Mask
![](https://i.imgur.com/o9TEHJ7.png)
- Normal Mask
![](https://i.imgur.com/vlH89Zs.png)

### 2-4) Hard Mining
Difficult train dataset which predicted incorrectly several times is collected in csv file.
Net is trained with combination of difficult train dataset and original train dataset.


# 3. Train
```
train.py
```

### 3-1) Optimizer 
[Stochastic Gradient Descent Optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)


### 3-2) Loss Function
Binary Cross Entropy Loss (torch.nn.BCELoss)


### 3-3) Hyperparameter

    patch size = 304

    normal threshold = 0.1  # normal mask inclusion ratio that select normal patches
    tumor threshold = 0.8   # tumor mask inclusion ratio that select tumor patches
    tissue threshold = 0.4  # tisse mask inclusion ratio that select tissue patches

    default learning rate = 0.005  # defalut learning ratio
    momentum = 0.9      # SGD optimizer parameter, 'momentum'
    weight decay = 5e-4 # SGD optimizer parameter, 'weight_decay'
    

# 4. Validation, Test

### 4-1) Statistical index
![](https://i.imgur.com/xewvt7l.png)
![](https://i.imgur.com/aaSab5K.png)


### 4-2) Checkpoint
- Info: Net, Accuracy, Loss, Recall, Specificity, Precision, F1_score,AUC, epoch, learning rate, threshold
    

# 5. Result

- First trial
![](https://i.imgur.com/LOFysOe.png)

- Second trial
![](https://i.imgur.com/nD8k8s3.png)

# 6. HeatMap
```
eval.py
```
- HeatMap Example

![](https://i.imgur.com/PvEVs8f.png)

# 7. Requirement
- [torch](http://pytorch.org/docs/master/nn.html)
- [torchvision](http://pytorch.org/docs/master/torchvision/transforms.html?highlight=torchvision%20transform)
- [openslide](http://openslide.org/api/python/)
- [opencv](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
- [argparse](http://pytorch.org/docs/0.3.0/notes/cuda.html?highlight=argparse)
- [matplotlib](https://matplotlib.org/)
- etcs..


# 8. Usage
1) Download the image zip files in camelyon17.
2) To create the dataset, run the utils.py.
3) Using the patches, train the model in train.py.
4) Run the eval.py.

# 9. Reference
- [camelyon17](https://camelyon17.grand-challenge.org/results/)
- [Resnet](https://arxiv.org/pdf/1512.03385.pdf)
- [Densenet](https://arxiv.org/pdf/1608.06993v5.pdf)

