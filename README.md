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

1. **prepare_dataset**: Extract frames, detect persons, and perform pose estimation using YOLO and Detectron2
2. **SAM3Dbody**: 3D body mesh reconstruction and keypoint extraction from video frames
3. **triangulation**: Triangulate 3D joint positions from multi-view 2D keypoints
4. **split_cycle**: Segment motion sequences into individual cycles using DTW and feature analysis
5. **videopose3d**: 3D pose estimation using temporal convolution networks
6. **camera_calibration**: Calibrate multi-camera setups for accurate 3D reconstruction
7. **analysis**: Data analysis and visualization tools

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
- Required model checkpoints (YOLO, SAM-3D-Body, etc.)

## Usage

The project uses Hydra for configuration management. Each module can be run independently:

### 1. Prepare Dataset (Pose Detection)
Extract keypoints and bounding boxes from videos using YOLO and Detectron2:

```bash
cd prepare_dataset
python main.py
```

Configuration: `configs/prepare_dataset.yaml`

### 2. SAM-3D-Body (3D Mesh Reconstruction)
Generate 3D body meshes and keypoints from video frames:

```bash
cd SAM3Dbody
python main.py
```

Configuration: `configs/sam3d_body.yaml`

### 3. Triangulation (3D Pose from Multi-view)
Triangulate 3D joint positions from synchronized multi-view videos:

```bash
cd triangulation
python main.py
```

Configuration: `configs/triangulation.yaml`

### 4. Split Cycle (Motion Segmentation)
Segment continuous motion into individual cycles:

```bash
cd split_cycle
python main.py
```

### 5. Camera Calibration
Calibrate multi-camera setup:

```bash
cd camera_calibration
python main.py
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
│   ├── prepare_dataset.yaml
│   ├── sam3d_body.yaml
│   └── triangulation.yaml
│
├── prepare_dataset/      <- Video preprocessing and keypoint extraction
│   ├── main.py           <- Main entry point
│   └── process/          <- Processing modules
│
├── SAM3Dbody/           <- 3D body mesh reconstruction
│   ├── main.py          <- Main entry point
│   ├── infer.py         <- Inference logic
│   └── sam_3d_body/     <- Model implementation
│
├── triangulation/       <- 3D pose triangulation from multi-view
│   ├── main.py          <- Main entry point
│   ├── load.py          <- Data loading utilities
│   └── vis/             <- Visualization tools
│
├── split_cycle/         <- Motion cycle segmentation
│   └── main.py          <- Cycle detection and splitting
│
├── videopose3d/         <- Temporal 3D pose estimation
│   ├── run.py           <- Training and inference
│   └── common/          <- Common utilities
│
├── camera_calibration/  <- Multi-camera calibration
│   └── main.py          <- Calibration pipeline
│
└── analysis/            <- Analysis and visualization notebooks
    └── load.ipynb       <- Data loading examples
```

## Configuration

All modules use Hydra for configuration management. Configuration files are located in the `configs/` directory.

Key configuration parameters:

- **paths**: Input/output paths for data and results
- **model**: Model checkpoints and parameters
- **YOLO**: Detection and tracking settings
- **camera_K**: Camera intrinsic parameters
- **camera_position**: Multi-camera setup geometry

## Key Technologies

- **PyTorch & PyTorch Lightning**: Deep learning framework
- **Hydra**: Configuration management
- **YOLOv11**: Person detection and pose estimation
- **Detectron2**: Instance segmentation and keypoint detection
- **SAM-3D-Body**: 3D body mesh reconstruction
- **VideoPose3D**: Temporal 3D pose estimation
- **OpenCV**: Video processing and computer vision
- **DTW (Dynamic Time Warping)**: Motion synchronization

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
