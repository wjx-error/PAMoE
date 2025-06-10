# PAMoE
> This repository provides the official implementation for CVPR 2025 paper 
> ["Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images"](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Learning_Heterogeneous_Tissues_with_Mixture_of_Experts_for_Gigapixel_Whole_CVPR_2025_paper.html).

We provide the implementation of PAMoE and demo implementations for integrating into [TransMIL](https://github.com/szc19990412/TransMIL) and vanilla Transformer.
The code implementation of the PAMoE plugin is located in `/models/pamoe_layers`.
The integration code for TransMIL and Transformer is in `/models/models_demo`.

<img src="/figs/PAMOE.jpg"/>

In this work, we introduce a plug-and-play **P**athology-**A**ware **M**ixture-**o**f-**E**xperts module (**PAMoE**), which identifies and processes tissue-specific features in the MoE layer, effectively tackling heterogeneous pathology tissues. PAMoE does not require specialized model workflow design and additional priors during inference, as it learns to route appropriate patches to its corresponding expert during training and discard patches that are irrelevant to the task.
We integrated PAMoE into various established WSI analysis methods and conducted experiments on the survival prediction task. 
The experimental results show that most transformer-based methods incorporated with PAMoE demonstrated performance improvements.

## Getting Start
### Data Preparation
#### Download the WSIs
The WSIs can be found in [TCGA](https://www.cancer.gov/tcga).

#### Patch Extraction
We follow [CLAM](https://github.com/mahmoodlab/CLAM) to cut whole slide images (WSIs) into patches (size $256\times 256$ at $20\times$ magnification),
and then generate the instance-level features using the [UNI](https://huggingface.co/MahmoodLab/UNI) encoder.
In the subsequent steps, we follow CLAM's storage format to obtain the patch coordinates and features.

#### Patch Classification
We obtain patch categories using the zero-shot classifier with [CONCH](https://github.com/mahmoodlab/CONCH).

#### Prototype 
After the patch extraction, 
users can obtain patch types by first edit the args configurations in `/data_preparation/classfication_CONCH.py`,
then run the following command
```
python /data_preparation/classfication_CONCH.py
```


Subsequently, prototypes are computed using the patch features and categories.
Users can obtain patch types by first edit the args configurations in `/data_preparation/make_prototypes_classifier.py`,
then run the following command
```
python /data_preparation/make_prototypes_classifier.py
```

We provide precomputed prototypes in folder `/prototypes/`.
We additionally provide a rapid prototype acquisition choice based on clustering in `/data_preparation/make_prototypes_cluster.py`,
the code is based on [PANTHER](https://github.com/mahmoodlab/PANTHER).

### Training
#### Make Splits
We provide the dataset lists and labels used for experiments in the /dataset_csv/ folder.
Users can split the dataset's case list into five folds for cross-validation.

```
python cross_val_split/TCGA_make_splits.py
```

#### Train
The configurations yaml file is in `/configs/`. 
Users may first modify the respective config files for hyper-parameter settings, and update the path to training config in train.py,
then run the following command to start train.

```
python train.py
```

## Acknowledgement
This work is supported by National Natural Science Foundation of China (Grant No. 82302316 and 62471133). This work is also supported by the Big Data Computing Center of Southeast University.
The PAMoE implementation is based on: 
[swiss-ai](https://github.com/swiss-ai/MoE),
[SwitchTransformers](https://github.com/kyegomez/SwitchTransformers),
and [flaxformer](https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py#L647-L717).
We would like to thank them for their contributions.

## Citing `PAMoE`

if you find `PAMoE` useful in your work, please cite our
[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Learning_Heterogeneous_Tissues_with_Mixture_of_Experts_for_Gigapixel_Whole_CVPR_2025_paper.pdf):

    @InProceedings{Wu_2025_CVPR,
        author    = {Wu, Junxian and Chen, Minheng and Ke, Xinyi and Xun, Tianwang and Jiang, Xiaoming and Zhou, Hongyu and Shao, Lizhi and Kong, Youyong},
        title     = {Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images},
        booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
        month     = {June},
        year      = {2025},
        pages     = {5144-5153}
    }



