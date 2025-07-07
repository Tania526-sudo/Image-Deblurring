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

## Project Structure
Image-Deblurring/
├── models/             # Model definitions
├── src/                # Core logic and helper functions
├── tests/              # Unit and integration tests
├── deblur.py           # Entry point to run deblurring
├── evaluate.py         # PSNR / SSIM evaluation
├── utils.py            # Utilities (I/O, preprocessing, metrics)
├── requirements.txt    # Python dependencies (pip)
├── pyproject.toml      # Project metadata (poetry)
├── .gitignore          # Ignored files and folders
├── README.md           # Project documentation
└── License             # License for use and distribution

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