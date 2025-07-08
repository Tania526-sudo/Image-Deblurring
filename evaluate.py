import os
import argparse
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

def compute_metrics(gt_path, pred_path):
    gt = imread(gt_path)
    pred = imread(pred_path)

    psnr_value = psnr(gt, pred, data_range=255)
    ssim_value = ssim(gt, pred, multichannel=True, data_range=255)

    return psnr_value, ssim_value

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deblurring Model")
    parser.add_argument('--gt_dir', type=str, required=True, help="Ground truth images directory")
    parser.add_argument('--pred_dir', type=str, required=True, help="Predicted images directory")
    args = parser.parse_args()

    gt_files = natsorted(glob(os.path.join(args.gt_dir, '*.*')))
    pred_files = natsorted(glob(os.path.join(args.pred_dir, '*.*')))

    if len(gt_files) != len(pred_files):
        raise ValueError("Mismatch in number of images between GT and Predictions")

    total_psnr = 0
    total_ssim = 0
    n = len(gt_files)

    for gt_path, pred_path in zip(gt_files, pred_files):
        p, s = compute_metrics(gt_path, pred_path)
        total_psnr += p
        total_ssim += s
        print(f"{os.path.basename(gt_path)} - PSNR: {p:.2f}, SSIM: {s:.4f}")

    print(f"\nAverage PSNR: {total_psnr/n:.2f}")
    print(f"Average SSIM: {total_ssim/n:.4f}")

if __name__ == '__main__':
    main()