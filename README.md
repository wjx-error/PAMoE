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

## Quick Plug-and-Play
PAMoE can be directly employed to replace the MLP layers within the models. 
The `capacity_factor` parameter serves to regulate the proportion of tokens discarded by PAMoE.
It is recommended that when setting capacity_factor to a relatively large value (e.g., 2), 
multiple PAMoE layers can be stacked (e.g., used consecutively or intermittently across multiple Transformer blocks).
Conversely, when capacity_factor is set to a smaller value (e.g., 1 or below), PAMoE is advised to be used as a standalone filter. 
Specific application scheme should be considered in conjunction with practical scenarios.

All PAMoE implementations are located in `/models/pamoe_layers`. 
Below is a simplified sample of embedding PAMoE into TransformerEncoder and TransformerEncoderBlock. 

The complete code can be found in `/models/models_demo/model_transformer_pamoe.py`. 

``` python
import torch
import torch.nn as nn
from models.pamoe_layers.pamoe_utils import drop_patch_cal_ce, get_x_cos_similarity, FeedForwardNetwork
from models.pamoe_layers.pamoe import PAMoE

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 # transformerblock settings
                 ...,
                 # PAMoE settings
                 use_pamoe=True, pamoe_use_residual=True, ...
                 ):
    ...
    self.pamoe_use_residual = pamoe_use_residual
    if use_pamoe:
        self.ffn = PAMoE(...)
    else:
        self.ffn = FeedForwardNetwork(...) # vanilla ffn
    ...
    
    def forward(self, x, ...):
        ...
        x_res = x
        x, gate_scores = self.ffn(x) # output x and gate scores os PAMoE
        x_index = x.clone() # used for remove zero tokens
        if self.pamoe_use_residual: # residual
            x = x_res + x
        ...
        return x, gate_scores, x_index
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 # transformer settings
                 layer_type=['pamoe', 'ffn', 'pamoe', 'ffn'], ...,
                 # PAMoE settings
                 drop_zeros=True, pamoe_use_residual=True, prototype_pth='./BRCA.pt', ...
                 ):
        ...
        layer_list = []
        for tp in layer_type:
            if tp.lower() == 'ffn':
                layer_list.append(TransformerEncoderBlock(use_pamoe=False, ...))
            elif tp.lower() == 'pamoe': # PAMoE layer
                layer_list.append(TransformerEncoderBlock(use_pamoe=True, pamoe_use_residual=pamoe_use_residual, ...))
        self.layers = nn.ModuleList(layer_list)
        self.proto_types = torch.load(prototype_pth, map_location='cpu') # load prototypes
        ...
        
    def forward(self, x, ...):
        # extract prototype probabilities
        similarity_scores = get_x_cos_similarity(x2, ..., self.proto_types)
        ...
        pamoe_loss_list = []
        gate_scores_list = []
        for layer in self.layers:  # transformer blocks
            x, gate_scores_tmp, x_index = layer(x, ...)
            # calculate PAMoE loss and drop zeros
            pamoe_loss_tmp, x, similarity_scores = drop_patch_cal_ce(drop_zeros=self.drop_zeros,...)
            pamoe_loss_list.append(pamoe_loss_tmp)
            gate_scores_list.append(gate_scores_tmp)
        logits = self.head(x)
        ...
        # The PAMoE loss should be weighted added to the main loss for backpropagation.
        loss_pamoe = torch.stack(pamoe_loss_list).mean()
        return gate_scores_list, logits, loss_pamoe
```
If you only want to simply replace the Feed-Forward Network (FFN) for experiments, 
you can refer to the following implementation.
``` python
import torch
import torch.nn as nn
from models.pamoe_layers.pamoe_utils import drop_patch_cal_ce, get_x_cos_similarity, FeedForwardNetwork
from models.pamoe_layers.pamoe import PAMoE
class TransformerEncoderBlock(nn.Module):
    def __init__(self, drop_zeros=True, prototype_pth='./BRCA.pt', ...):
        # self.ffn = FeedForwardNetwork(...) # vanilla ffn
        self.ffn = PAMoE(...)
        self.proto_types = torch.load(prototype_pth, map_location='cpu') # load prototypes
        
    def forward(self, x, ...):
        # extract prototype probabilities
        similarity_scores = get_x_cos_similarity(x, ..., self.proto_types)
        # PAMoE layer, and the residual connection can be optionally adopted here.
        x, gate_scores = self.ffn(x)
        # calculate PAMoE loss and drop zeros
        pamoe_loss, x, similarity_scores = drop_patch_cal_ce(drop_zeros=self.drop_zeros,...)
        ...
        logits = self.head(x)
        return gate_scores, logits, pamoe_loss
```
### Explanation of the Parameters `drop_zeros` and `pamoe_use_residual`
In [Expert Choice MoE](https://arxiv.org/pdf/2202.09368v2), 
certain tokens deemed less relevant during inference may not be routed to any expert. 
Without explicit intervention, these tokens would be output as zero vectors.

Given the ubiquity of residual connections in Transformer architectures, 
our empirical studies demonstrate that directly incorporating these zero-valued residuals into the main pathway can lead to performance degradation.

Thus, we provide the `drop_zeros` and `pamoe_use_residual` Parameters.

When the sorting relationship between tokens is meaningful, like [TransMIL](https://github.com/szc19990412/TransMIL),
we suggest disabling both residual connections and zero-padding removal `drop_zeros=False, pamoe_use_residual=False`. 
This preserves the original sequence structure while mitigating the adverse effects of zero residuals.

When the sorting relationship is not important (e.g., in ViT, positional encoding has been used to mark the positions of tokens),
we suggest enabling zero-token dropping and retaining residual connections `drop_zeros=True, pamoe_use_residual=True`. 

The configuration `drop_zeros=False, pamoe_use_residual=True` represents the conventional approach of directly integrating all residuals, including zero vectors.

For the simplest plug-and-play, 
the parameter `drop_zeros` should be retained and directly passed into the `drop_patch_cal_ce` method. 
Whether to use residual connections should be controlled and implemented within the forward function by yourself.

## Getting Start
### Data Preparation
#### Download the WSIs
The WSIs can be found in [TCGA](https://www.cancer.gov/tcga).

#### Patch Extraction
We follow [CLAM](https://github.com/mahmoodlab/CLAM) to cut whole slide images (WSIs) into patches (size $256\times 256$ at $20\times$ magnification),
and then generate the instance-level features using the [UNI](https://huggingface.co/MahmoodLab/UNI) encoder.
In the subsequent steps, we follow CLAM's storage format (.h5 files) to obtain the patch coordinates and features as model inputs.

#### Patch Classification and Prototype Extraction
After the patch extraction, we obtain patch categories using the zero-shot classifier with [CONCH](https://github.com/mahmoodlab/CONCH).
Users can obtain patch types by first edit the args configurations in `/data_preparation/classfication_CONCH.py`,
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

We provide extracted prototypes in folder `/prototypes/`.
We additionally provide a rapid prototype extraction choice based on clustering in `/data_preparation/make_prototypes_cluster.py`,
the cluster based code is based on [PANTHER](https://github.com/mahmoodlab/PANTHER).

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
Our primary inspiration comes from [Expert Choice MoE](https://arxiv.org/pdf/2202.09368v2).
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



