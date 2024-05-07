# FullRot_WRMix
The official implementation of FullRot + WRMix provides a comprehensive set of functions for deep learning and computer vision tasks. It includes training process printing, log saving, and computation and storage of standard metrics for classification, semantic segmentation, and object detection tasks.

[**Any region can be perceived equally and effectively on rotation pretext task using full rotation and weighted-region mixture**](https://ieeexplore.ieee.org/document/10230242)  
Wei Dai, Tianyi Wu, Rui Liu, Min Wang, Jianqin Yin, Jun Liu        
Accepted in Neural Networks, 2024. [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4495231)][[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024002740?dgcid=rss_sd_all)]


![](./readme/framework.svg)

![](./readme/mixcam.svg)

## Installation

For instructions on how to install the FullRot_WRMix implementation, please refer to the [INSTALL.md](readme/INSTALL.md) file.

## Benchmark and Evaluation

To prepare the datasets required for benchmark evaluation and training, please consult the [DATA.md](readme/DATA.md) file. 

For pre-training the resnet50 model using self-supervised learning, you can follow the settings provided in [main.sh](shell/main.sh) script. Additionally, we provide downstream task training settings in the [downstream_example.sh](shell/downstream_example.sh) script.

### *Classification results on STL-10, CIFAR-10/100, Sports-100, Mammals-45, and PAD-UFES-20*  

<p align="left"> <img src=readme/classification.png align="center" width="1080px">

### *Semantic segmentation results on PASCAL VOC 2012, ISIC 2018, FloodArea, and TikTokDances*  

<p align="left"> <img src=readme/segmentation.png align="center" width="720px">

### *Object Detection results on PASCAL VOC 2007 and UTDAC 2020*  

<p align="left"> <img src=readme/detection.png align="center" width="720px">

### Ablation study
For the training settings of the ablation study and extracurricular experiments, please refer to the [ablation.sh](shell/ablation.sh) and [ablation_extra.sh](shell/ablation_extra.sh) scripts.

## Citation

If you find this implementation useful in your research, we kindly request that you consider citing our paper as follows:

    @article{dai2024any,
      title={Any region can be perceived equally and effectively on rotation pretext task using full rotation and weighted-region mixture},
      author={Dai, Wei and Wu, Tianyi and Liu, Rui and Wang, Min and Yin, Jianqin and Liu, Jun},
      journal={Neural Networks},
      pages={106350},
      year={2024},
      publisher={Elsevier}
    }
