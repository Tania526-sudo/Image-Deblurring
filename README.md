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
│
├── src/ # Core source code (models, utils, training)
├── tests/ # Unit tests and evaluation scripts
├── README.md # Project overview
├── pyproject.toml # Project configuration and dependencies
└── .gitignore # Ignored files

## Installation

```bash
git clone https://github.com/Tania526-sudo/Image-Deblurring.git
cd Image-Deblurring
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

Or with poetry (if using pyproject.toml):
poetry install