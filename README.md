# Occlusion-Aware and Consistency-Enhanced 3D Gaussian Splatting for Crowdsourced Heritage Reconstruction


This repository contains the official authors implementation associated with the paper "Occlusion-Aware and Consistency-Enhanced 3D Gaussian Splatting for Crowdsourced Heritage Reconstruction".

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:ning112106/OACE_3DGS.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/ning112106/OACE_3DGS.git --recursive
```
The components have been tested on Ubuntu Linux 20.04. Instructions for setting up and running each of them are in the below sections.


## Setup

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements
- 12 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+
- OpenGL 4.5-ready GPU and drivers

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate OACE_3DGS
```

### External Dependencies
This project relies on the following external models: YOLOv11 (for Occlusion Detection) and LaMa (for Image Inpainting)

#### YOLOv11 (for Occlusion Detection)
Since we need to configure the YOLO detection classes and generate the corresponding masks, we have uploaded branch ultralytics to the repository. For detailed installation instructions, please refer to [YOLO11](https://github.com/ultralytics/ultralytics/tree/new-v11-head).The main installation steps are as follows:
```shell
git clone https://github.com/ning112106/OACE_YOLOv11.git --recursive
conda create -n yolo11 python=3.9
conda activate yolo11
pip install ultralytics
```

#### LaMa (for Image Inpainting)
For detailed installation instructions, please refer to [LaMa](https://github.com/advimman/lama).The main installation steps are as follows:
```shell
git clone https://github.com/advimman/lama.git --recursive
conda env create -f conda_env.yml
conda activate lama
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
pip install pytorch-lightning==1.2.9
```

## Datasets preparation
- Download the Photo Tourism Datasets (We use three scenes: Brandenburg gate, Trevi fountain, and Sacre coeur in our experiments) from [Image Matching Challenge PhotoTourism (IMC-PT) 2020 dataset](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
- Download the NeRF-OSR Datasets (We use three scenes: europa, lwp, and st in our experiments) from [NeRF-OSR Dataset](https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk?dir=/Data).

#### Preprocessing
- Pedestrian Detection and Occlusion Mask Generation

```shell
conda activate yolo11
cd ultralytics
python mask.py --source <path to scene>/images --save-path <path to scene>/masks
```
In the implementation process, we use yolo11x.pt as the weight file. If you want to change the detection category, you can modify classes in predict.py

- Image Inpainting

```shell
conda activate lama
cd lama
curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
mv <path to scene>/masks <path to scene>/images <path to scene>/image_blend

python3 bin/predict.py \
refine=True \
model.path=<path to big-lama>/big-lama/ \
indir=<path to scene>/image_blend \
outdir=<path to scene>/inpainteds \
model.checkpoint=best.ckpt
```
- Generate the Overlapping Dictionaries 
```shell
conda activate OACE_3DGS
cd OACE_3DGS/tools

python generate_overlap_dict.py \
    --colmap_path <path to scene>/sparse/0_txt \
    --output <path to scene>/overlap_dict.json \
    --min_common 50 \
    --min_overlap_ratio 0.2
```

<br>
<details>
<summary><span style="font-weight: bold;">The Tree Structure of Each Dataset</span></summary>

```
brandenburg_gate/
├── images
├── inpainteds
├── masks
├── nb-info.json
├── overlap_dict.json
└── sparse
    ├── 0
    │   ├── cameras.bin
    │   ├── images.bin
    │   ├── points3D.bin
    │   ├── points3D.ply
    │   ├── test.txt
    │   └── train.txt
    ├── 0_txt
    │   ├── cameras.txt
    │   ├── frames.txt
    │   ├── images.txt
    │   ├── points3D.txt
    │   └── rigs.txt
    └── points3D.ply
```
</details>

## Running

### Training

To run the optimizer, simply use

```shell
python train.py -s <path to scene>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>
To train them while withholding a test set for evaluation, use the ```--eval``` flag as follows:

```shell
python train.py -s <path to scene> -m <path to trained model> --eval # Train with train/test split
```

### Rendering

```shell
python render.py -m <path to trained model> # Generate renderings
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

### Evaluating

```shell
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.

</details>
<br>

## Acknowledgments

Our code is based on the awesome Pytorch implementation of 3D Gaussian Splatting (3DGS). The authors thank the anonymous reviewers for their valuable feedback. We appreciate all the contributors. 






