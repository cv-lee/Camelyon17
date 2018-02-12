# 1. Camelyon17 #


## 1-1) About
This project is the CNN model to diagnose breast cancer. It consists of a CNN model,
Desenet and ResNet. The criteria for performance is AUC(Area Under the Curve). 
An output of model is a float number from 0 to 1. (0: Normal, 1: Tumor)


## 1-2) Architecture
![](https://i.imgur.com/7yH9SKm.jpg)




# 2. Dataset
[camelyon 17 dataset](https://camelyon17.grand-challenge.org/)


## 2-1) Overview
[CAMELYON17](https://camelyon17.grand-challenge.org/) is the second grand challenge in pathology organised 
by the Diagnostic Image Analysis Group (DIAG) and Department of Pathology of the Radboud University Medical 
Center (Radboudumc) in Nijmegen, The Netherlands.
The data in this challenge contains whole-slide images (WSI) of hematoxylin and eosin (H&E) stained lymph node sections.
All ground truth annotations were carefully prepared under supervision of expert pathologists. For the purpose of revising the slides, 
additional slides stained with cytokeratin immunohistochemistry were used. If however, you encounter problems 
with the dataset, then please report your findings at the forum.


## 2-2) Data Argumentation

Convert Images to Horizontal Flip
Convert Images to Vertical Flip
Convert Images to Gray Scale Randomly (percentage = 10%)
Convert Images brightness, contrast, saturation, hue slightly 


## 2-3) Mask


## 2-4) Hard Mining



# 3. Train

## 3-1) Optimizer 
[Stochastic Gradient Descent Optimizer](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)


## 3-2) Loss Function
Binary Cross Entropy Loss (torch.nn.BCELoss)


## 3-3) Hyperparameter



# 4. Validation, Test



# 5. HeatMap


# 6. Requirement
[torch](http://pytorch.org/docs/master/nn.html)

[torchvision](http://pytorch.org/docs/master/torchvision/transforms.html?highlight=torchvision%20transform)

[os](http://www.pythonforbeginners.com/os/pythons-os-module)

[argparse](http://pytorch.org/docs/0.3.0/notes/cuda.html?highlight=argparse)

[csv](https://docs.python.org/3/library/csv.html?highlight=csv#module-contents)

[matplotlib](https://matplotlib.org/)


# 7. Usage


# 8. Reference



