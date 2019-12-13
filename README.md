# BraTs   

[TOC]



### 1. Overview

<div align="center">
  <img src="https://i.imgur.com/emAFrL1.gif">  <img src="https://i.imgur.com/dGrmh2x.gif">
  <br>
  <em align="center">Fig 1: Brain Complete Tumor Segmention</em>
  <br>
  <img src="https://i.imgur.com/n0WAMwh.gif">  <img src="https://i.imgur.com/PFTwmVb.gif">
  <br>
  <em align="center">Fig 2: Brain Core Tumor Segmention</em>
  <br>
  <br>
  <img src="https://placehold.it/15/1589F0/000000?text=+">
  <em align="center">Ground Truth</em>
  <br>
  <img src="https://placehold.it/15/f03c15/000000?text=+">
  <em align="center">Prediction</em>
  <br>
</div>

BRaTS stands for Brain Tumor Segmentation. The [BRaTS](http://braintumorsegmentation.org/) challenge has always been focusing on the evaluation of the state-of-the-art methods for the segmentation of brain tumors in multi-modal magnetic resonance imaging (MRI) scans. This is a coordinated effort for Tumor Segmentation from the [University of Pennsylvania, Perelman School of Medicine](https://www.med.upenn.edu/).

MRI Scans of Glioblastomas/High Grade Glioma (GBM/HGG) and low grade glioma (LGG) with pathologically confirmed diagnosis are labelled and are available for download.

### 2. Models 

  - **U-Net**

```bash
pytorch/models/unet.py
```

<div align="center">
  <img src="https://i.imgur.com/OXtVFvT.png">
  <br>
  <br>
  <em align="center">Fig 3: U-Net Diagram </em>
  <br>
</div>

 - **DeepLab V3 +**

```bash
tensorflow/models/research/deeplab
```

<div align="center">
  <img src="https://i.imgur.com/5IBKzDx.png">
  <br>
  <br>
  <em align="center">Fig 5: DeepLab V3+ Diagram </em>
  <br>
</div>

### 3. Dataset

##### 3.1 Overview

File: FLAIR MRI Sequence Data of One Person

File Type: png files

 Image-Shape: *240(Slide Width) × 240(Slide Height) × 31(Number of Slide) × 1(Multi-mode)*

Image Subjects: 31 persons

##### 3.2 Labels

- GAD-Enhancing Tumor - WHITE
- Tumor Core - BLACK
- Whole Tumor - GREY
- Background - GREYISH BLACK

##### 3.3 Data Preprocessing

- Co-registering
- Interpolation to the same resolution (1 mm^3)
- Skull Stripped



### 4. Train

##### 3.1 Loss Function
  [Dice Coefficient Loss](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

![](https://i.imgur.com/aGUbIeU.png)

##### 4.2 Optimizer
  [Adam Optimizer](https://arxiv.org/pdf/1412.6980.pdf)

  [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  

##### 4.3 Hyperparameters
  learning rate = 1e-4

  maximum number of epochs = 1000

  Weights Init: Normal Distribution (mean:0, std:0.01)

  Bias Init: Initialized as 0

### 5. Steps to Run



**Step 1**: Download complete model and unzip from [here]( https://drive.google.com/open?id=1_Rfrnq97S9clrDdGSGsOO0KN1ETluWqy).

**Step 2**: 

```
git clone https://github.com/geekswaroop/BRaTS.git
```

cd into the downloaded folder and then run

**Step 3:**

```
python train.py
```

### 6. Results

![](/home/krishna/Coding/Intelligence/BRaTS/results.png)

### 7. References

1) [The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](https://ieeexplore.ieee.org/document/6975210)

2) [Investigator's Implementation](https://github.com/JooHyun-Lee/BraTs)

3) [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

