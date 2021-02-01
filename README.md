# Stable Head Pose Estimation and Landmark Regression via 3D Dense Face Reconstruction

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

## Face Reconstruction

Our network is multi-task since it can directly regress 3DMM params from a single face image for reconstruction, as well as estimate the head pose via **R** and **T** prediction.
During training, the predicted **R** can supervise the params regression branch to generate refined face mesh.
Meanwhile, the landmarks calculated via the params are provided as the labeled data to the pose estimation branch for training, through the `cv2.solvePnP` tool.

Theoretically speaking, the head pose estimation task in our model is weakly supervised, since only a few labeled data is required to activate the training process.
In detail, the loss function at the initial stage can be described as follows:

<p align="center">
  <img alt="Init Pose Loss" src="https://latex.codecogs.com/svg.latex?L_{pose}=L_{2}(R_{gt},R_{pose})+L_{2}(R_{gt},R_{params})">
</p>

After the initialization process, the ground truth can be replaced by the prediction results of other branches.
Therefore, the loss function will be transformed to the following equation in the later stage of the training process:

<p align="center">
  <img alt="Pose Loss" src="https://latex.codecogs.com/svg.latex?L_{pose}=2\cdot%20L_{2}(R_{params},R_{pose})">
</p>

In general, fitting the 3D model during the training process dynamically can avoid the inaccurate head pose estimation results caused by the coarse predefined model.

### Head Pose

<p align="center">
  <img alt="pose demo" src="https://s3.ax1x.com/2021/01/14/sdfSJI.gif">
</p>

Traditional head pose estimation approaches, such as [Appearance Template Models](https://www.researchgate.net/publication/2427763_Face_Recognition_by_Support_Vector_Machines), [Detector Arrays](https://ieeexplore.ieee.org/document/609310), and [Mainfold Embedding](https://ieeexplore.ieee.org/document/4270305) have been extensively studied.
However, methods based on deep learning improved the prediction accuracy to meet actual needs, until recent years.

Given a set of predefined 3D facial landmarks and the corresponding 2D image projections, the SolvePnP tool can be utilized to calculate the rotation matrix.
However, the adopted mean 3D human face model usually introduces intrinsic error during the fitting process.
Meanwhile, the additional landmarks extraction component is also kind of cumbersome.

Therefore, we designed a network branch for directly regress **6DoF** parameters from the face image.
The predictions include the 3DoF rotation matrix **R** (Pitch, Yaw, Roll), and the 3DoF translation matrix **T** (x, y, z),
Compared with the landmark-based method, directly regression the camera matrix is more robust and stable, as well as significantly reduce the network training cost.
Run the demonstrate script in **pose** mode to view the real-time head pose estimation results:

``` bash
python3 demo_video.py -m pose -f <your-video-path>
```

### Expression

![Expression](https://s3.ax1x.com/2021/01/06/sZV0BQ.jpg)

Coarse expression estimation can be achieved by combining the predefined expressions in BFM linearly.
In our model, regression the **E** is one of the tasks of the params prediction branch.
Obviously, the accuracy of the linear combination is positively related to the dimension.

Clipping parameters can accelerate the training process, however, it can also reduce reconstruction accuracy, especially details such as eye and mouth.
More specifically, **E** has a greater impact on face details than **S** when emotion is involved.
Therefore, we choose 10-dimension for a tradeoff between the speed and the accuracy, the training data can be found at [here](https://github.com/cleardusk/3DDFA#training-details) for refinement.

In addition, we provide a simple facial expression rendering script. Run the following command for illustration:

``` bash
python3 demo_image.py <your-image-path>
```

### Mesh

<p align="center">
  <img alt="mesh demo" src="https://s3.ax1x.com/2021/01/30/ykCWEd.gif">
</p>

According to the predefined BFM and the predicted 3DMM parameters, the dense 3D facial landmarks can be easily calculated.
On this basis, through the index mapping between the morphable triangle vertices and the dense landmarks defined in BFM, the renderer can plot these geometries with depth infomation for mesh preview.
Run the demonstrate script in **mesh** mode for real-time face reconstruction:

``` bash
python3 demo_video.py -m mesh -f <your-video-path>
```

## Benchmark

Our network can directly output the camera matrix and sparse or dense landmarks.
Compared with the model in the original paper with the same backbone, the additional parameters yield via the pose regression branch does not significantly affect the inference speed, which means it can still be CPU real-time.

| Scheme | THREAD=1 | THREAD=2 | THREAD=4 |
| :-: | :-: | :-: | :-: |
| Inference  | 7.79ms  | 6.88ms | 5.83ms |

In addition, since most of the operations are wrapped in the model, the time consumption of pre-processing and post-processing are significantly reduced.
Meanwhile, the optimized lightweight renderer is 5x faster (3ms vs 15ms) than the [Sim3DR](https://github.com/cleardusk/3DDFA_V2/tree/master/Sim3DR) tools.
These measures decline the latency of the entire pipeline.

| Stage | Preprocess | Inference | Postprocess | Render |
| :-: | :-: | :-: | :-: | :-: |
| Each face cost  | 0.23ms  | 7.79ms | 0.39ms | 3.92ms |

Run the following command for speed benchmark:

``` bash
python3 video_speed_benchmark.py <your-video-path>
```

## Citation

``` bibtex
@inproceedings{guo2020towards,
    title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author={Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020}
}
```
