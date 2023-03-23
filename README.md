# MaLP: Manipulation Localization Using a Proactive Scheme
Official Pytorch implementation of CVPR 2023 paper "MaLP: Manipulation Localization Using a Proactive Scheme ".

[Vishal Asnani](https://github.com/vishal3477), [Xi Yin](https://xiyinmsu.github.io/), [Tal Hassner](https://talhassner.github.io/home/), [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/overview_4.png?raw=true)
## Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2

## Getting Started

## Datasets 
- Every GM is used with different datasets they are trained on. The GM-dataset information is given in Tab. 2 of the supplmentary. Please refer to the test images released by [Proactive detection work](https://github.com/vishal3477/proactive_IMD). 
- For other GMs, we will release the test images soon. 
- The training data is used as CELEBA.

## Pre-trained model
The pre-trained model trained on STGAN can be downloaded using the information below:

Model     | Link 
---------|--------
Localization only | [Model]()    
Localization + Detection | [Model]()    

## Training
- Go to the folder STGAN
- Download the STGAN repository files and pre-trained model from https://github.com/csmliu/STGAN
- Provide the train and test path in respective codes as sepecified below. 
- Provide the model path to resume training
- Run the code as shown below:

```
python train.py
```



## Testing using pre-trained models
- Download the repository files and pre-trained model of GMs in the respective folder, StarGAN: https://github.com/yunjey/stargan , CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix , GauGAN: https://github.com/NVlabs/SPADE
- Download the pre-trained model for our template from https://drive.google.com/file/d/1p9zETa9rCU0wx8wD5Ige2TbCL8WciV7o/view?usp=sharing
- Provide the model path in the code
- Run the code as shown below for StarGAN:

```
python test_stargan.py
```
- Run the code as shown below for CycleGAN:

```
python test_cyclegan.py
```
- Run the code as shown below for GauGAN:

```
python test_gaugan.py
```


If you would like to use our work, please cite:
```
@inproceedings{asnani2022pro_loc
      title={MaLP: Manipulation Localization Using a Proactive Scheme}, 
      author={Asnani, Vishal and Yin, Xi and Hassner, Tal and Liu, Xiaoming},
      booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
      
}
```
