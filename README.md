# Camelyon17 #


## About
It is a project to diagnose breast cancer. It consists of a CNN model,
Desenet121. The criteria is AUC(Area Under the Curve). Also, It configures the
thresholding optimization.  


## Architecture
![](/home/intern/Desktop/DenseBlock.png/)


# Dataset
[camelyon 17 dataset](https://camelyon17.grand-challenge.org/)


## Optimizer 
SGDoptimizer 


## Requirement
[torch](http://pytorch.org/docs/master/nn.html)

[torchvision](http://pytorch.org/docs/master/torchvision/transforms.html?highlight=torchvision%20transform)

[os](http://www.pythonforbeginners.com/os/pythons-os-module)

[argparse](http://pytorch.org/docs/0.3.0/notes/cuda.html?highlight=argparse)

[csv](https://docs.python.org/3/library/csv.html?highlight=csv#module-contents)

[matplotlib](https://matplotlib.org/)


## Usage


## CNN Architecture
```
python models.py
```


## Train and Test
By selecting only specific slides, the ratio of tumor to normal was divided by 1:6.

```
python dataset.py
python main.py
```


## Utility
Tumor mask was extracted using xml. Tissues that do not know the coordinates
values are made a mask using HSV and Ostu. 
After saving normal patches and tumor patches, we chose them separately.
we also considered that tumor patches must not include in normal patches.

To speed up, we attempted to pretrained. In addition, by saving the wrong patches, we have implemented hard mining. 

```
python utils.py
```

##Reference



