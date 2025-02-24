# PPCC-CD: Cross-Domain Plant Point Cloud Completion Based on Feature Low-Rank Mapping and Dual Frequency Prompts for Plant Phenotyping Analysis
The official implementation of the paper：
PPCC-CD: Cross-Domain Plant Point Cloud Completion Based on Feature Low-Rank Mapping and Dual Frequency Prompts for Plant Phenotyping Analysis

Contact: xiaomengli@cau.edu.cn Any questions or discussion are welcome!

-----
+ [2025.12.31] We have uploaded the dataset, which can be downloaded from the following link: https://drive.google.com/file/d/1NjSIlX1obYQ0HNbU5oZRrEwV3z1pA1bX/view?usp=sharing.

+ [2025.12.31] We have initialized the repo. The related resources will be released after the manuscript is accepted.


<img src="assets/PPCC-CD.png" alt="Dataset" width="800" height="600">




## Abstract
In real-world growth environments, 3D plant point cloud data collected are often incomplete due to environmental interference, occlusions, and sensor limitations. This incompleteness poses significant challenges to the development of plant phenotypic analysis, growth monitoring, and digital twin applications. Existing plant point cloud completion
methods often neglect the potential value of complete point clouds collected from other environments, leaving cross-domain data underutilized. Efficiently leveraging such data is essential for improving completion performance under data-constrained conditions. To address these challenges, this paper proposes a cross-domain plant point cloud completion method that integrates a Feature Low-Rank Mapping (FLRM) module and a Dual-Frequency Domain Prompt (DFDP) module. The FLRM module reduces cross￾domain distribution discrepancies by mapping features between the source and target domains into a shared low-rank space, thereby enhancing completion performance in the target domain. The DFDP module, based on Graph Fourier Transform, explicitly
guides the model to learn geometric features more accurately, further improving completion precision. Experimental results on the proposed PPCC-CD dataset demonstrate that the method significantly enhances the accuracy and robustness of cross-domain
point cloud completion, providing a novel technical pathway for efficient plant point cloud reconstruction and phenotypic analysis.

## Contributions
1. A cross-domain plant point cloud completion model, PPCD-CD, is introduced to fully leverage source domain data for completing missing points in the target domain. The model demonstrates robust performance, particularly in scenarios with limited target domain data, ensuring high-quality completion even under challenging conditions.
   
2. The Feature Low-Rank Mapping module is designed to reduce the distribution gap between the source and target domains. By employing low-rank mapping, it 
facilitates a shared feature space across domains, thereby enhancing cross-domain completion performance.

3. Dual-Frequency Domain Prompt is proposed to extract both global and local frequency domain features through the Graph Fourier Transform. This module explicitly guides the model in capturing critical structural information, enabling more 113
precise recognition of the overall morphology of plant point clouds and significantly improving completion quality.

4. We construct the first cross-domain plant point cloud dataset, which encompasses diverse plant morphologies and various acquisition environments. This dataset provides a valuable resource for advancing research on cross-domain point cloud completion. The datasets are available at https://github.com/liqingque/PPCC-CD.


`

## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.




### Inference

To inference sample(s) with pretrained model

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```


### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```


### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
## Acknowledgement
A large part of the code is borrowed from [Anchorformer](https://github.com/chenzhik/AnchorFormer), [PoinTr](https://github.com/ifzhang/ByteTrack), [P2C] Thanks for their wonderful works!

## Citation
The related resources will be released after the manuscript is accepted. 
