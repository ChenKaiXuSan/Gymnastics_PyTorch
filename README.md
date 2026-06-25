<div align="center">    
 
# Gymnastics PyTorch - 3D Pose Estimation Pipeline     

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

</div>
 
## Description   
A comprehensive PyTorch-based pipeline for 3D human pose estimation and motion analysis from multi-view gymnastics videos. This project integrates multiple state-of-the-art models and techniques to process gymnastics videos through detection, pose estimation, 3D reconstruction, and motion cycle segmentation.

### Key Features
- **Multi-view 3D Pose Estimation**: Reconstruct 3D poses from synchronized multi-camera views
- **SAM-3D-Body Integration**: 3D body mesh reconstruction and keypoint extraction
- **Motion Cycle Segmentation**: Automatically segment gymnastics movements into cycles
- **Camera Calibration**: Support for multi-camera setup calibration
- **Pose Tracking**: Person detection and tracking across video frames
- **Triangulation**: 3D joint position estimation from 2D multi-view keypoints

## Pipeline Overview

The project consists of several interconnected modules that form a complete 3D motion analysis pipeline:

1. **SAM3Dbody**: 3D body mesh reconstruction and keypoint extraction from video frames
2. **fuse**: Align and fuse SAM3D-Body results from face and side views
3. **split_cycle**: Segment motion sequences into individual cycles using DTW and feature analysis
4. **project/train**: Train and evaluate motion classification models
5. **analysis**: Data analysis and visualization tools
6. **triangulation**: Legacy/support path for 3D joint position estimation from multi-view 2D keypoints
7. **camera_calibration**: Calibrate multi-camera setups for accurate 3D reconstruction

## Installation

First, clone the repository and install dependencies:

```bash
# Clone the project
git clone https://github.com/ChenKaiXuSan/Gymnastics_PyTorch.git
cd Gymnastics_PyTorch

# Install the package
pip install -e .

# Install required dependencies
pip install -r requirements.txt
```

### Additional Requirements
- Python 3.6+
- PyTorch (latest version)
- CUDA (for GPU acceleration)
- Required model checkpoints (SAM-3D-Body, etc.)

## Usage

The project uses Hydra for configuration management. The active data preparation
path uses SAM3D-Body directly. See `docs/current_pipeline.md` for the current
pipeline and `docs/modules.md` for module responsibilities.

### 1. SAM-3D-Body (Dataset Preparation)
Generate 3D body meshes and keypoints from video frames:

```bash
python -m SAM3Dbody.main
```

Configuration: `configs/sam3d_body.yaml`

The previous YOLO/Detectron2/DPT/RAFT preprocessing flow is kept under
`legacy/prepare_dataset/` with configuration in `configs/legacy/prepare_dataset.yaml`.

### 2. Fuse Multi-View Results
Align and fuse SAM3D-Body outputs from face and side views:

```bash
python -m fuse.main
```

### 3. Split Cycle (Motion Segmentation)
Segment continuous motion into individual cycles:

```bash
python -m split_cycle.main
```

### 4. Train Classifiers
Train and evaluate motion classification models:

```bash
python -m project.train.train
```

Configuration: `configs/train.yaml`

### 5. Optional Support Modules
Triangulation and camera calibration are kept as support workflows:

```bash
python -m triangulation.main
```

Configuration: `configs/triangulation.yaml`

Calibrate multi-camera setup:

```bash
python -m camera_calibration.main
```

## Project Organization

```txt
├── README.md              <- The top-level README for developers
├── requirements.txt       <- Python dependencies
├── setup.py              <- Makes project pip installable
├── LICENSE               <- Apache 2.0 License
│
├── configs/              <- Hydra configuration files
│   ├── config.yaml
│   ├── inference.yaml
│   ├── sam3d_body.yaml
│   ├── train.yaml
│   ├── triangulation.yaml
│   └── legacy/
│       └── prepare_dataset.yaml
│
├── SAM3Dbody/           <- 3D body mesh reconstruction
│   ├── main.py          <- Main entry point
│   ├── infer.py         <- Inference logic
│   └── sam_3d_body/     <- Model implementation
│
├── legacy/
│   └── prepare_dataset/  <- Previous YOLO/Detectron2/DPT/RAFT preprocessing flow
│
├── triangulation/       <- 3D pose triangulation from multi-view
│   ├── main.py          <- Main entry point
│   ├── load.py          <- Data loading utilities
│   └── vis/             <- Visualization tools
│
├── split_cycle/         <- Motion cycle segmentation
│   └── main.py          <- Cycle detection and splitting
│
├── fuse/                <- Multi-view SAM3D-Body result alignment and fusion
│   └── main.py          <- Main fusion entry point
│
├── project/             <- Training and cross-validation package
│   ├── train/           <- Models, dataloaders, trainers, evaluation
│   └── cross_validation/ <- Fold/index generation and CV entry points
│
├── videopose3d/         <- Temporal 3D pose estimation
│   ├── run.py           <- Training and inference
│   └── common/          <- Common utilities
│
├── camera_calibration/  <- Multi-camera calibration
│   └── main.py          <- Calibration pipeline
│
├── analysis/            <- Analysis and visualization scripts/notebooks
│
├── docs/                <- Current pipeline and module documentation
│
└── tests/               <- Lightweight checks and tests
```

## Configuration

All modules use Hydra for configuration management. Configuration files are located in the `configs/` directory.

Key configuration parameters:

- **paths**: Input/output paths for data and results
- **model**: Model checkpoints and parameters
- **model**: SAM3D-Body and training model parameters
- **camera_K**: Camera intrinsic parameters
- **camera_position**: Multi-camera setup geometry

## Key Technologies

- **PyTorch & PyTorch Lightning**: Deep learning framework
- **Hydra**: Configuration management
- **SAM-3D-Body**: 3D body mesh reconstruction
- **VideoPose3D**: Temporal 3D pose estimation
- **OpenCV**: Video processing and computer vision
- **DTW (Dynamic Time Warping)**: Motion synchronization

YOLOv11, Detectron2, DPT, and RAFT remain available only in the legacy
`legacy/prepare_dataset/` preprocessing flow.

## Requirements

Main dependencies (see `requirements.txt` for complete list):
- pytorch-lightning
- torch & torchvision
- ultralytics (YOLOv11)
- hydra-core
- opencv-python
- numpy
- scikit-learn
- transformers
- tensorboard

## Data Format

The pipeline processes data in the following formats:

- **Input**: Multi-view video files (.mp4, .mov, .avi, etc.)
- **Intermediate**: PyTorch tensor files (.pt) with keypoints, bboxes, and features
- **Output**: 3D joint positions (.npz), segmented video clips, 3D visualizations

## Docker Support

Docker configuration is available in the `docker/` directory for containerized deployment.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

**Kaixu Chen**  
The University of Tsukuba  
Email: chenkaixusan@gmail.com

## Acknowledgments

This project builds upon several excellent open-source projects:
- [YOLOv11](https://github.com/ultralytics/ultralytics) for object detection
- [Detectron2](https://github.com/facebookresearch/detectron2) for pose estimation
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) for temporal pose estimation
- SAM-3D-Body for 3D mesh reconstruction

### Citation   
If you use this project in your research, please cite:
```bibtex
@misc{gymnastics_pytorch,
  title={Gymnastics PyTorch: 3D Pose Estimation Pipeline},
  author={Chen, Kaixu},
  year={2026},
  publisher={GitHub},
  url={https://github.com/ChenKaiXuSan/Gymnastics_PyTorch}
}
```
