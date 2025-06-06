# PAMoE
> This repository provides the official implementation for CVPR 2025 paper ["Learning Heterogeneous Tissues with Mixture of Experts for Gigapixel Whole Slide Images"](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Learning_Heterogeneous_Tissues_with_Mixture_of_Experts_for_Gigapixel_Whole_CVPR_2025_paper.pdf).

**Due to the rush of graduation, we haven't had enough time to organize all the code yet.**

**For now, we provide the implementation of PAMoE and demo implementations for integrating into [TransMIL](https://github.com/szc19990412/TransMIL) and vanilla Transformer.**

The code implementation of the PAMoE plugin is located in `/models/pamoe_layers`.
The integration code for TransMIL and Transformer is also in `/models/models_demo`.

<span style="color:red">**The complete code will be released soon.**</span>

<img src="/figs/PAMOE.jpg"/>

In this work, we introduce a plug-and-play **P**athology-**A**ware **M**ixture-**o**f-**E**xperts module (**PAMoE**), which identifies and processes tissue-specific features in the MoE layer, effectively tackling heterogeneous pathology tissues. PAMoE does not require specialized model workflow design and additional priors during inference, as it learns to route appropriate patches to its corresponding expert during training and discard patches that are irrelevant to the task.
We integrated PAMoE into various established WSI analysis methods and conducted experiments on the survival prediction task. 
The experimental results show that most transformer-based methods incorporated with PAMoE demonstrated performance improvements.

## Acknowledgement
This work is supported by National Natural Science Foundation of China (Grant No. 82302316 and 62471133). This work is also supported by the Big Data Computing Center of Southeast University.
The PAMoE implementation is based on: [flaxformer](https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py#L647-L717),
[swiss-ai](https://github.com/swiss-ai/MoE), and [SwitchTransformers](https://github.com/kyegomez/SwitchTransformers).
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



