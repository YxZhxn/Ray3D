# Ray3D: ray-based 3D human pose estimation for monocular absolute 3D localization

This repository contains the implementation of the approach described in the paper:
> Yu Zhan, Fenghai Li, Renliang Weng, and Wongun Choi. 
Ray3D: ray-based 3D human pose estimation for monocular absolute 3D localization. 
In Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

###### 3D pose estimation by Ray3D in the world coordinate system
<p align="center"><img src="images/ray3d.h36m.gif" width="100%" alt="" /></p>

# Quick start

### Dependencies
Please make sure you have the following dependencies installed before running:
* python 3
* torch==1.4.0
* other necessary dependencies [`requirements.txt`](requirements.txt)
* (optional) screen, rsync

### Dataset

##### Human3.6M
We use the data processed by [Videopose](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md). 
You can generate the files by yourself as well, 
or you can download them from 
[google drive](https://drive.google.com/drive/folders/1Md_mwtkACG3VF0JY5Usx5lzL-s68ispz?usp=sharing).

##### MPI-INF-3DHP
3DHP is set up by our own [script](data/prepare_data_3dhp.py). 
You can download the original [dataset](http://vcai.mpi-inf.mpg.de/3dhp-dataset/) 
and generate the files with the following command:
```bash
# set up the 'data_root' parameter which stores the original 3DHP data
python3 prepare_data_3dhp.py
```
Or you can directly download the processed data from 
[google drive](https://drive.google.com/drive/folders/1Md_mwtkACG3VF0JY5Usx5lzL-s68ispz).

##### HumanEva-I
We set up HumanEva-I by following the [procedure](data/prepare_data_humaneva.py) provided by 
[Videopose](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md). 
You can set it up by yourself, or you can download the files from 
[google drive](https://drive.google.com/drive/folders/1Md_mwtkACG3VF0JY5Usx5lzL-s68ispz?usp=sharing).

##### Synthetic
The synthetic dataset is set up based on Human3.6M. Once you have the 'data_3d_h36m.npz' file generated,
you can generate the synthetic dataset with following procedure:
```bash
# 1). generate synthetic data for camera intrinsic test
python3 camera_intrinsic.py
```
Then, run the following preprocessing script:
```bash
# 2). generate synthetic data for camera extrinsic test
python3 camera_intrinsic.py
```

Finally, use the following preprocessing script to generate training file for synthetic training
```bash
# 3). generate train and evaluation file for synthetic training
python3 aggregate_camera.py
```

### Description
We train and test five approaches on the above mentioned datasets.
* Ray3D: implemented in the [`main`](https://github.com/YxZhxn/Ray3D/tree/main) branch.
* RIE: implemented in the [`main`](https://github.com/YxZhxn/Ray3D/tree/main) branch.
* Videopose: implemented in the [`videopose`](https://github.com/YxZhxn/Ray3D/tree/videopose) branch.
* Poseformer: implemented in the [`poseformer`](https://github.com/YxZhxn/Ray3D/tree/poseformer) branch.
* Poselifter: implemented in the [`poselifter`](https://github.com/YxZhxn/Ray3D/tree/poselifter) branch.

We release the [pretrained models](https://drive.google.com/drive/folders/1YTYJc6Y9CUG4U7HuZZe5l7tyT145BxxN?usp=sharing) 
for academic purpose. You can create a folder named `checkpoint` to store all the pretrained models.

### Train
Please turn on `visdom` before you start training a new model. 

To train the above mentioned methods, you can run the following command by 
specifying different configuration file in the `cfg` folder:
```bash
python3 main.py --cfg cfg_ray3d_3dhp_stage1
```

To train Ray3D with synthetic data, please use the codes from 
[`synthetic`](https://github.com/YxZhxn/Ray3D/tree/synthetic) branch. 
We did some optimization for large scale training.

### Evaluation
To evaluate the models with public and synthetic datasets, 
you can run the following command by specifying different configuration files,
timestamps and checkpoints:
```bash
python3 main.py \
    --cfg cfg_ray3d_h36m_stage3 \
    --timestamp Oct_31_2021_05_43_36 \
    --evaluate best_epoch.bin
```

### Visualization
We use the same visualization techniques provided by 
[VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
You can perform visualization with the following command:
```bash
python3 main.py \
    --cfg cfg_ray3d_h36m_stage3 \
    --timestamp Oct_31_2021_05_43_36 \
    --evaluate best_epoch.bin \
    --render
```

### Citation
If you find this repository useful, please cite our paper:
```
@Inproceedings{yzhan2022,
  Title          = {Ray3D: ray-based 3D human pose estimation for monocular absolute 3D localization},
  Author         = {Yu Zhan, Fenghai Li, Renliang Weng, and Wongun Choi},
  Booktitle      = {CVPR},
  Year           = {2022}
}
```

### Acknowledgement
Our implementation took inspiration from the following authors and repositories:
* [matteorr](https://github.com/matteorr)
* [Pose3D-RIE](https://github.com/paTRICK-swk/Pose3D-RIE)
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [PoseFormer](https://github.com/zczcwh/PoseFormer)
* [PoseLifter](https://github.com/juyongchang/PoseLifter)
* [SPIN](https://github.com/nkolot/SPIN)
* [human36m-camera-parameters](https://github.com/karfly/human36m-camera-parameters)

We thank the authors for kindly releasing their codes!