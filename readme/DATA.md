# Dataset preparation

## STL-10 Dataset
The STL-10 dataset is a general image recognition dataset designed for unsupervised learning inspired by the CIFAR-10 dataset. 

- To obtain the dataset, download it from [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/).

- After downloading the dataset, you need to rearrange the binary files containing images and labels using the `tools/stl10Save.py` script.

- The desired output data structure should be as follows:
~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- STL10_unlabelled
    `-- |--- train
            |--- unlabelled_img_1.png
            |--- unlabelled_img_2.png
            |--- ...
            |--- unlabelled_img_27100.png
            |--- ...
~~~

~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- STL10
    `-- |--- train
            |--- airplane_1.png
            |--- airplane_2.png
            |--- ...
            |--- car_262.png
            |--- ...
        |--- test
            |--- airplane_1.png
            |--- airplane_2.png
            |--- ...
            |--- bird_45.png
            |--- ...
        |--- ...
~~~

## CIFAR-10/100 Dataset
The CIFAR-10 and CIFAR-100 datasets are labelled subsets of the 80 million tiny images dataset.

- To obtain the dataset, you can download it from [CIFAR-10 dataset](https://www.kaggle.com/c/cifar-10/) and [CIFAR-100 dataset](https://www.kaggle.com/datasets/fedesoriano/cifar100).

- After downloading, you need to rearrange the images with the same class (each class uses a folder with the class name).

- The desired output data structure should be as follows:
~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- CIFAR-10/100
  `-- |--- train
    `-- |--- airplane/apple
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
    `-- |--- automobile/aquarium_fish
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
    `-- |--- ...
            |--- ...
    `-- |--- truck/train
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
  `-- |--- test
    `-- |--- airplane/apple
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
    `-- |--- automobile/aquarium_fish
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
    `-- |--- ...
            |--- ...
    `-- |--- truck/train
            |--- 0001.png
            |--- 0002.png
            |--- ...
            |--- 0026.png
            |--- ...
~~~

## PAD-UFES-20 Dataset

The data of PAD-UFES-20 is a smartphone skin lesion image. 

- To obtain the dataset, you can download it from [PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones](https://data.mendeley.com/datasets/zr7vgbcyr2/1).

- After downloading, you need to rearrange images with the same class (each class uses a folder with the class name).

- Create the five split subsets for cross-validation.

- The desired output data structure should be as follows:

~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- PAD2020S5
    `-- |--- split1
            |--- ack_1.jpg
            |--- ack_2.jpg
            |--- ...
            |--- bcc_1.jpg
            |--- ...
        |--- split2
            |--- ack_1.jpg
            |--- ack_2.jpg
            |--- ...
            |--- bcc_1.jpg
            |--- ...
        |--- ...
~~~

## PASCAL VOC 2012 Dataset
The PASCAL VOC 2012 dataset is a realistic scene image recognition dataset. 

- To obtain the dataset, you can download it from [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

- After downloading, rearrange the images and masks within the same folder.

- The desired output data structure should be as follows:
~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- PASCALVOC
    `-- |--- train
            |--- 2008_000015.jpg
            |--- 2008_000019.jpg
            |--- ...
            |--- 2011_003238.jpg
            |--- ...
        |--- val
            |--- 2008_000009.jpg
            |--- 2008_000016.jpg
            |--- ...
            |--- 2011_003085.jpg
            |--- ...
        |--- mask
            |--- 2008_000009.png
            |--- 2008_000015.png
            |--- ...
            |--- 2011_003238.png
            |--- ...
~~~

## ISIC 2018 Dataset
The data of the ISIC 2018 dataset is skin lesion data with corresponding binary masks. 

- To obtain the dataset, you can download it from [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/#2018).

- After downloading, rearrange the images and masks within the same folder.

- The desired output data structure should be as follows:
~~~
${FullRot_WRMix_ROOT}
|-- dataset
`-- |-- PASCALVOC
    `-- |--- train
            |--- ISIC_0000000.jpg
            |--- ISIC_0000001.jpg
            |--- ...
            |--- ISIC_0013319.jpg
            |--- ...
        |--- val
            |--- ISIC_0012255.jpg
            |--- ISIC_0012346.jpg
            |--- ...
            |--- ISIC_0036291.jpg
            |--- ...
        |--- mask
            |--- ISIC_0000000.png
            |--- ISIC_0000001.png
            |--- ...
            |--- ISIC_0012346.png
            |--- ...
~~~

## PASCAL VOC 2007 Dataset
Follows the detectron2 settings in the [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets).

## Sports-100, Mammals-45, FloodArea, TikTokDances, and UTDAC 2020
Please refer to their official websites.

Sports-100: [100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification).

Mammals-45: [Mammals Image Classification Dataset (45 Animals)](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals).

FloodArea: [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).

TikTokDances: [Human Segmentation Dataset - TikTok Dances](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset).

UTDAC 2020: [Boosting-R-CNN](https://github.com/mousecpn/Boosting-R-CNN-Reweighting-R-CNN-Samples-by-RPN-s-Error-for-Underwater-Object-Detection).


## References

If you use the datasets and our data pre-processing codes, we kindly request that you consider citing our paper as follows:

~~~
@inproceedings{coates2011analysis,
  title={An analysis of single-layer networks in unsupervised feature learning},
  author={Coates, Adam and Ng, Andrew and Lee, Honglak},
  booktitle={Proceedings of the fourteenth international conference on artificial intelligence and statistics},
  pages={215--223},
  year={2011},
  organization={JMLR Workshop and Conference Proceedings}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Toronto, ON, Canada}
}

@article{pacheco2020pad,
  title={PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones},
  author={Pacheco, Andre GC and Lima, Gustavo R and Salomao, Amanda S and Krohling, Breno and Biral, Igor P and de Angelo, Gabriel G and Alves Jr, F{\'a}bio CR and Esgario, Jos{\'e} GM and Simora, Alana C and Castro, Pedro BC and others},
  journal={Data in brief},
  volume={32},
  pages={106221},
  year={2020},
  publisher={Elsevier}
}

@misc{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}

@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic)},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and Celebi, M Emre and Dusza, Stephen and Gutman, David and Helba, Brian and Kalloo, Aadi and Liopyris, Konstantinos and Marchetti, Michael and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}

@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018},
  publisher={Nature Publishing Group}
}

@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}	


@article{dai2023any,
    title={Any Region Can Be Perceived Equally and Effectively on Rotation Pretext Task Using Full Rotation and Weighted-Region Mixture},
    author={Dai, Wei and Wu, Tianyi and Liu, Rui and Wang, Min and Yin, Jianqin and Liu, Jun},
    year={2023}
}
~~~
