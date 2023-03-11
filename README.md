# Leveraging Self-Supervised Learners in Classifying Aerial Imagery

### Introduction
With the broad adoption of commercial drones, collecting aerial imagery is trivial. Turning it into a form that’s useful is not. Machine learning has been changing the calculus, but the most modern and powerful methods require labeled data. The best computer vision algorithms are all supervised learners, but recent advancements in self-supervised learning have narrowed this gap. The intent of this work was to investigate how the self-supervised approaches SimCLR, MoCo, and BYOL could be helpful in classifying aerial crop imagery of rice seedlings.


| SimCLR | MoCo | BYOL |
| :---: | :---: | :--: |
| ![SimCLR](ReadmeImages/Screenshot%202023-03-03%20at%204.19.39%20PM.png) | ![MoCo](ReadmeImages/Screenshot%202023-03-03%20at%204.22.56%20PM.png) | ![BYOL](ReadmeImages/Screenshot%202023-03-03%20at%204.23.55%20PM.png) |

\* *Figures are from their respective papers.*

### The Dataset
Rice is cultivated on every continent except Antartica, and half of the world's population eats rice every day. Between 1961 and 2019, rice yields have increased by a factor of 2.5, with the highest yields occurring in developed regions. The worldwide significance of rice cultivation, and the role that cost-effective yield estimation plays in precision agriculture underpinned the motivation to use a UAV-collected aerial imagery dataset of rice seedlings. This dataset consisted of two classes, rice shoots and arable land. 


![https://github.com/aipal-nchu/RiceSeedlingDataset#1-data-download-link](ReadmeImages/Screenshot%202023-03-03%20at%204.52.27%20PM.png)
From: https://github.com/aipal-nchu/RiceSeedlingDataset#1-data-download-link

## How to run:
The script was made to run locally on Apple M-series GPU/MPS, but will fall back to running on CPU. It's a significant speed-up running on MPS vs CPU. **To run on MPS, make sure your version of Pytorch supports this.** At the time of this writing, this was only available with the nightly version of Pytorch:

General pytorch install instructions:
https://pytorch.org/get-started/locally/

MacOS Pytorch Nightly install for MPS support:
```zsh
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

Clone the repository:
```zsh
git clone https://github.com/dmyklejord/selfsupervised-learning-for-image-classification.git
```

Script can be run with:
```zsh
python3 main.py
```
The script will start fine-tuning a ResNet-18 for 10-epochs using the MoCo-V2 framework.

## Results
### Using the linear classification protocol:
| Algorithm | Details | Epochs | Seedling Data Accuracy | Cifar-10 Accuracy |
| --- | --- | --- | --- | --- |
| Baseline | ImageNet Pretrained ResNet-18 | 0 | 0.9900 | 0.733 |
| SimCLR | Training From Random | 50 | 0.9992 | 0.715 |
| SimCLR | Fine-Tuning | 1 | 0.9987 | 0.806 |
| SimCLR | Fine-Tuning | 10 | 0.9997 | 0.830 |
| MoCo | Training From Random | 50 | 0.9976 | 0.637 |
| MoCo | Fine-Tuning | 1 | 0.9977 | 0.816 |
| MoCo | Fine-Tuning | 10 | 0.9991 | 0.832 |
| BYOL | Training From Random | 50 | 0.9915 | 0.644 |
| BYOL | Fine-Tuning | 1 | 0.9974 | 0.711 |
| BYOL | Fine-Tuning | 10 | 0.9991 | 0.817 |

### TSNE Visualization of SimCLR Seedling data:
![SimCLR TSNE](ReadmeImages/Screenshot%202023-03-03%20at%207.56.02%20PM.png)

Explore the interactive TSNE visualizations by opening the .html files in the InteractivePlots folder.

### Training Resources (On Apple M1 Macbook Pro, 16GB/256GB):
| Algorithm | Seedling Data Minutes per Epoch | Cifar-10 Minutes per Epoch | Seedling Data Memory Usage (GB) | Cifar-10 Memory Usage |
| :---: | :---: | :--: | :---: | :---: |
| SimCLR | 21 | 24 | 8.5 | 8.5 |
| MoCo-V2 | 15 | 17 | 5.5 | 5.5 |
| BYOL | 25 | 28 | 8.5 | 8.5 |


## Running on Custom Data:
I've only tested using .tif images, but the documentation
seems to allow other formats (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)


The "data" folder should contain:

data/
- class_x
  - xxx.tif
  - xxy.tif
  - ...
  - xxz.tif
- class_y
  - 123.tif
  - nsdf3.tif
  - ...
  - asd932_.tif

In the case the data is unlabeled, use:

data/
- dataset_name
  - xxx.tif
  - xxy.tif
  - ...
  - xxz.tif


---
---

## What is self-supervised learning?
Image classification is typical done using supervised algorithms—that is, algorithms that utilize a large dataset of images with corresponding ground-truth labels for each image. These supervised systems learn vector representations of the images’ features, that can then be used for downstream tasks such as classification or clustering into the different classes. The large cost of gathering and curating such large, labeled, datasets has spurred research in alternative ways to learn image feature representations. In the absence of labeled image data, this learning can be done using self-supervised deep-learning algorithms (“SSL”) that use the **unlabeled data itself as a supervisory signal** during training.

This is a good explaination using SimCLR as an example: https://amitness.com/2020/03/illustrated-simclr/

