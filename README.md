# Image Deblurring with Neural Networks

This project implements and evaluates deep learning techniques for **image deblurring** using Python and PyTorch. The goal is to restore sharp images from blurred inputs caused by motion or defocus, which is critical in computer vision pipelines for medical imaging, surveillance, and photography.

---

## Features

- CNN-based deblurring architecture
- Synthetic blur generation (Gaussian, Motion)
- Training and evaluation pipeline
- PSNR / SSIM metrics for model quality
- Modular structure for reproducible research

---
<pre> ```bash
## Project Structure
Image-Deblurring/
├── notebooks/                 # Jupyter notebooks for inference and evaluation
│   ├── Deblur_Inference.ipynb
│   └── Deblur_Evaluation.ipynb
├── OpenVINO/                  # Scripts for converting the model to OpenVINO and running inference
│   ├── convert_to_openvino.py
│   ├── openvino_inference.py
│   └── model_deblurring.onnx  # Converted model in ONNX format for OpenVINO
├── pretrained_models/         # Pre-trained model weights
│   └── model_deblurring.pth
├── results/                   # Saved model results
├── src/                      
├── tests/                    
├── utils/                    # Auxiliary utilities
│   ├── __init__.py
│   ├── dataset_utils.py       # Functions for working with datasets
│   ├── dir_utils.py           # Functions for working with directories
│   ├── image_utils.py         # Image processing and conversion
│   └── model_utils.py         # Auxiliary functions for models
├── deblur.py                 # Main script to start de-blurring
├── evaluate.py               # Script for evaluating the quality of results (PSNR, SSIM)
├── get-pip.py                # Script to install dependencies (optional)
├── logger.py                 # Logging processes and results
├── MPRNet.py                 # Implementation of the MPRNet model architecture
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata for Poetry
├── README.md                 # Project documentation
└── License                   # License
``` </pre>
## Installation
Clone the repository and install dependencies using either pip or poetry.
Option 1: Using pip
git clone https://github.com/Tania526-sudo/Image-Deblurring.git
cd Image-Deblurring

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Option 2: Using Poetry
poetry install
