# Learning-spall-failure-with-Neural-Operators

This repository contains the code and data for the ML part of the paper:

Numerical and data-driven modeling of spall failure in polycrystalline ductile materials

Indrashish Saha and Lori Graham-Brady

![alt text](https://raw.githubusercontent.com/Indrashish95/Learning-spall-failure-with-Neural-Operators/main/media/30grain1_gif.gif)

Link to arxiv: [https://doi.org/10.48550/arXiv.2507.03706]

## Dataset

The dataset is hosted at: *link*

1. training data of microstructures with 30 grains [aspect ratio = 1]
2. test data:
   (a) 30 grains [aspect ratio = 1]
   (b) 30 grains [aspect ratio = 0.33]
   (c) 30 grains [aspect ratio = 3]
   (d) 10 grains [aspect ratio = 1]
   (e) 20 grains [aspect ratio = 1]
   (f) 50 grains [aspect ratio = 1]

## DL Models

There are 3 DL models used in the paper:


1. 3D U-net 

2. FNO 3D [based on implementation of https://github.com/CUBELeonwang/FNO-3DUM and https://github.com/neuraloperator/neuraloperator/tree/main]

3. U-FNO 3D [https://github.com/gegewen/ufno]

*All the codes are modified according to the need of the current problem and only takes the classes of FNO and U-FNO from these implementations*




