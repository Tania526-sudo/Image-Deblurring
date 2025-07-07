import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from glob import glob
from natsort import natsorted
import argparse
from skimage import img_as_ubyte
import cv2
from collections import OrderedDict

from MPRNet import MPRNet  


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully.")


def process_images(model, input_dir, result_dir, device):
    os.makedirs(result_dir, exist_ok=True)

    files = natsorted(glob(os.path.join(input_dir, '*.jpg'))
                      + glob(os.path.join(input_dir, '*.JPG'))
                      + glob(os.path.join(input_dir, '*.png'))
                      + glob(os.path.join(input_dir, '*.PNG')))
    if len(files) == 0:
        raise Exception(f"No image files found in {input_dir}")

    img_multiple_of = 8

    model.eval()
    model.to(device)

    with torch.no_grad():
        for file_ in files:
            img = Image.open(file_).convert('RGB')
            input_ = TF.to_tensor(img).unsqueeze(0).to(device)

            h, w = input_.shape[2], input_.shape[3]
            H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
            W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0

            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model(input_)
            restored = restored[0]
            restored = torch.clamp(restored, 0, 1)

            restored = restored[:, :h, :w]
            restored = restored.permute(1, 2, 0).cpu().numpy()
            restored = img_as_ubyte(restored)

            filename = os.path.splitext(os.path.basename(file_))[0]
            save_img(os.path.join(result_dir, f"{filename}_deblurred.png"), restored)
            print(f"Processed and saved: {filename}_deblurred.png")


def main():
    parser = argparse.ArgumentParser(description='Image Deblurring with MPRNet')
    parser.add_argument('--input_dir', required=True, type=str, help='Directory with blurred input images')
    parser.add_argument('--result_dir', required=True, type=str, help='Directory to save deblurred images')
    parser.add_argument('--weights', required=True, type=str, help='Path to pretrained model weights (.pth)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MPRNet()
    load_checkpoint(model, args.weights)

    process_images(model, args.input_dir, args.result_dir, device)


if __name__ == "__main__":
    main()