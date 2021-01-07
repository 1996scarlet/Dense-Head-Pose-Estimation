# Dense Head Pose Estimation: Towards Fast, Accurate and Stable 3D Dense Face Alignment

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
* (Cautious) For Windows user, please refer to [this tutorial](https://stackoverflow.com/questions/1130479/how-to-build-a-dll-from-the-command-line-in-windows-using-msvc) for more details.

## 3D Facial Landmarks

68--sparse

``` bash
python3 demo_video.py -m sparse -f <your-video-path>
```

38365--dense

``` bash
python3 demo_video.py -m dense -f <your-video-path>
```

## Head Pose and Expression

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

![HeadPoseLogo](https://s3.ax1x.com/2021/01/06/sZV0BQ.jpg)

## Citation

``` bibtex
@inproceedings{guo2020towards,
    title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author={Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2020}
}
```
