import os
import argparse
from glob import glob
from natsort import natsorted

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import img_as_ubyte
import cv2
from collections import OrderedDict
from runpy import run_path

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"Loaded model weights from {weights_path} on {device}")

def process_image(model, image_path, output_path, device, img_multiple_of=8):
    model.eval()
    
    with torch.no_grad():
        img = Image.open(image_path).convert('RGB')
        input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0

        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), mode='reflect')

        restored = model(input_tensor)
        restored = torch.clamp(restored[0], 0, 1)
        restored = restored[:, :h, :w]
        restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Важливо: squeeze перед permute
        restored_img = img_as_ubyte(restored)

        save_img(output_path, restored_img)
        print(f"Saved deblurred image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Image Deblurring with MPRNet")
    parser.add_argument('--input', required=True, type=str,
                        help="Path to blurred image or directory with images")
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Directory to save deblurred images")
    parser.add_argument('--weights', required=True, type=str,
                        help="Path to pretrained weights (.pth) file")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Завантажуємо модель з MPRNet.py, який має бути у тому ж каталозі
    load_file = run_path("MPRNet.py")
    model = load_file['MPRNet']()
    model.to(device)

    load_checkpoint(model, args.weights, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        files = natsorted(glob(os.path.join(args.input, '*.jpg'))
                          + glob(os.path.join(args.input, '*.png'))
                          + glob(os.path.join(args.input, '*.jpeg')))
        if not files:
            raise FileNotFoundError(f"No images found in {args.input}")
        for file_path in files:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(args.output_dir, f"{filename}_deblurred.png")
            process_image(model, file_path, output_path, device)
    else:
        filename = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output_dir, f"{filename}_deblurred.png")
        process_image(model, args.input, output_path, device)

if __name__ == "__main__":
    main()

