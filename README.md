# MaLP: Manipulation Localization Using a Proactive Scheme
Official Pytorch implementation of CVPR 2023 paper "MaLP: Manipulation Localization Using a Proactive Scheme ".

[Vishal Asnani](https://github.com/vishal3477), [Xi Yin](https://xiyinmsu.github.io/), [Tal Hassner](https://talhassner.github.io/home/), [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

[Paper](http://cvlab.cse.msu.edu/pdfs/asnani_yin_hassner_liu_cvpr2023.pdf) [Supplementary](http://cvlab.cse.msu.edu/pdfs/asnani_yin_hassner_liu_cvpr2023_supp.pdf)


![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/overview_4.png?raw=true)
## Prerequisites

- PyTorch 1.5.0
- Numpy 1.14.2
- Scikit-learn 0.22.2

## Getting Started

## Datasets 
- Every GM is used with different datasets they are trained on. The GM-dataset information is given in Tab. 2 of the supplmentary. Please refer to the test images released by [Proactive detection work](https://github.com/vishal3477/proactive_IMD). 
- For new datasets used, please go [here](https://drive.google.com/file/d/1JfOqxlGhbvVGdiTK5P47YywYMe8C1Eob/view?usp=drive_link). 
- The training data is used as CELEBA.

## Pre-trained model
The pre-trained model trained on STGAN can be downloaded using the information below:

Model     | Link 
---------|--------
Localization only | [Model](https://drive.google.com/file/d/1fIoiVpZMNtn_wr-yo8verYX8lQY30Zle/view?usp=share_link)    
Localization + Detection | [Model](https://drive.google.com/file/d/1bZGWG_TTN5Gers0V4VXDiGPq82Tc86qG/view?usp=sharing)    

## Training
- Install VIT transformer package following the instructions in https://github.com/lucidrains/vit-pytorch
- Download the STGAN repository files and pre-trained model from https://github.com/csmliu/STGAN and place the train_loc_det.py file in that folder
- Run the code as shown below:

```
python train_loc_det.py --data_train "YOUR DATA PATH" --resume --model_path "MODEL PATH" 
```

For training only localization module, run the code as shown below:
```
python train_loc.py --data_train "YOUR DATA PATH" --resume --model_path "MODEL PATH" 
```


## Testing using pre-trained models
- Download the pre-trained model using the above links. 
- Provide the model path in the code
- Run the code as shown below:

```
python evaluation_loc_det.py --data_train "YOUR DATA PATH" --resume --model_path "MODEL PATH" 
```

## Visualization
STGAN 
![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/visualization_supp_1.png?raw=true)

STGAN 
![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/visualization_supp_2.png?raw=true)

DRIT
![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/visualization_supp_3.png?raw=true)

GauGAN
![alt text](https://github.com/vishal3477/pro_loc/blob/main/images/visualization_supp_4.png?raw=true)




If you would like to use our work, please cite:
```
@inproceedings{asnani2023pro_loc
      title={MaLP: Manipulation Localization Using a Proactive Scheme}, 
      author={Asnani, Vishal and Yin, Xi and Hassner, Tal and Liu, Xiaoming},
      booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
      
}
```
