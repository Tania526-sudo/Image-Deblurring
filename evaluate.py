import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from deblur import load_model, preprocess

def evaluate_model(blur_dir, sharp_dir, model_name='mprnet', device='cpu'):
    model = load_model(model_name, device)
    model.eval()

    blur_images = sorted(glob(os.path.join(blur_dir, '*')))
    sharp_images = sorted(glob(os.path.join(sharp_dir, '*')))

    psnr_list, ssim_list = [], []

    for blur_path, sharp_path in zip(blur_images, sharp_images):
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')

        input_tensor = preprocess(blur_img).to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_img = output_tensor.squeeze(0).cpu().clamp(0,1).permute(1,2,0).numpy()
        output_img = (output_img * 255).astype(np.uint8)

        sharp_np = np.array(sharp_img)

        current_psnr = psnr(sharp_np, output_img, data_range=255)
        current_ssim = ssim(sharp_np, output_img, multichannel=True, data_range=255)

        psnr_list.append(current_psnr)
        ssim_list.append(current_ssim)

        print(f"{os.path.basename(blur_path)}: PSNR={current_psnr:.2f}, SSIM={current_ssim:.4f}")

    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate deblurring model on dataset")
    parser.add_argument('--blur_dir', required=True, help="Directory with blurred images")
    parser.add_argument('--sharp_dir', required=True, help="Directory with sharp images")
    parser.add_argument('--model', default='mprnet', choices=['mprnet'])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    evaluate_model(args.blur_dir, args.sharp_dir, args.model, args.device)