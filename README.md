# Stable Head Pose Estimation via 3D Dense Face Reconstruction

![FaceReconstructionDemo](https://s3.ax1x.com/2021/01/06/sZVyhq.gif)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/1996scarlet/Dense-Head-Pose-Estimation.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/1996scarlet/Dense-Head-Pose-Estimation/context:python)
[![License](https://badgen.net/github/license/1996scarlet/Dense-Head-Pose-Estimation)](LICENSE)
[![ECCV](https://badgen.net/badge/ECCV/2020/red)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3162_ECCV_2020_paper.php)

Reimplementation of [(ECCV 2020) Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://github.com/cleardusk/3DDFA_V2) via Tensorflow Lite framework, face mesh, head pose, landmarks, and more.

* CPU real-time face deteciton, alignment, and reconstruction pipeline.
* Lightweight render library, 5x faster (3ms vs 15ms) than the [Sim3DR](https://github.com/cleardusk/3DDFA_V2/tree/master/Sim3DR) tools.
* Camera matrix and dense/sparse landmarks prediction via a single network.
* Generate facial parameters for robust head pose and expression estimation.

## Setup

### Basic Requirements

* Python 3.6+
* `pip3 install -r requirements.txt`

### Render for Dense Face

* GCC 6.0+
* `bash build_render.sh`
* **(Cautious)** For Windows user, please refer to [this tutorial](https://stackoverflow.com/questions/1130479/how-to-build-a-dll-from-the-command-line-in-windows-using-msvc) for more details.

## 3D Facial Landmarks

In this project, we perform dense face reconstruction by 3DMM parameters regression.
The regression target is simplified as camera matrix (**C**, shape of 3x4), appearance parameters (**S**, shape of 1x40), and expression variables (**E**, shape of 1x10), with 62 dimensions in total.

The sparse or dense facial landmarks can be estimated by applying these parameters to a predefined 3D model, such as [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details).
More specifically, the following formula describes how to generate a face through parameters:

<p align="center">
  <img alt="Generate Face" src="https://latex.codecogs.com/svg.latex?F=U_{base}+S\cdot%20W_{shp}+E\cdot%20W_{exp}">
</p>

where **U** and **W** are from pre-defined face model.
Combine them linearly with parameters to generate sparse or dense faces.
Finally, we need to integrate the posture information into the result:

<p align="center">
  <img alt="With matrix" src="https://latex.codecogs.com/svg.latex?F=R\cdot%20F+T">
</p>

where **R** (shape of 3x3) and **T** (shape of 3x1) denote rotation and translation matrices, respectively, which are fractured from the camera matrix **C**.

### Sparse

<p align="center">
  <img alt="sparse demo" src="https://s3.ax1x.com/2021/01/10/slO9je.gif">
</p>

Since we have reconstructed the entire face, the 3D face alignment can be achieved by selecting the landmarks at the corresponding positions. See [[TPAMI 2017] Face alignment in full pose range: A 3d total solution](https://arxiv.org/abs/1804.01005) for more details.

Comparing with the method of first detecting 2D landmarks and then performing depth estimation, directly fitting 3DMM to solve 3D face alignment can not only obtain more accurate results in larger pose scenes, but also has obvious advantages in speed.

We provide a demonstration script that can generate 68 landmarks based on the reconstructed face. Run the following command to view the real-time 3D face alignment results:

``` bash
python3 demo_video.py -m sparse -f <your-video-path>
```

### Dense

<p align="center">
  <img alt="dense demo" src="https://s3.ax1x.com/2021/01/09/sQ01VP.gif">
</p>

Currently, our method supports up to 38,365 landmarks.
We draw landmarks every 6 indexes for a better illustration.
Run the demonstrate script in **dense** mode for real-time dense facial landmark localization:

``` bash
python3 demo_video.py -m dense -f <your-video-path>
```

## Head Pose and Expression

![Head Pose](https://s3.ax1x.com/2021/01/14/sdfSJI.gif)

``` bash
python3 demo_video.py -m pose -f <your-video-path>
```

![Expression](https://s3.ax1x.com/2021/01/06/sZV0BQ.jpg)

``` bash
python3 demo_image.py <your-image-path>
```

## Face Mesh Reconstruction

``` bash
python3 demo_video.py -m mesh -f <your-video-path>
```

| Scheme | THREAD=1 | THREAD=2 | THREAD=4 |
| :-: | :-: | :-: | :-: |
| Inference  | 7.79ms  | 6.88ms | 5.83ms |

``` bash
python3 video_speed_benchmark.py <your-video-path>
```

| Stage | Preprocess | Inference | Postprocess | Render |
| :-: | :-: | :-: | :-: | :-: |
| Each face cost  | 0.23ms  | 7.79ms | 0.39ms | 3.92ms |

## Citation

``` bibtex
@inproceedings{guo2020towards,
    title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author={Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020}
}
```
