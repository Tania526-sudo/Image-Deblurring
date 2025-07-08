import os
import argparse
from openvino.runtime import Core
import cv2
import numpy as np
from natsort import natsorted
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)        # batch dimension
    return img

def postprocess_output(output):
    output = np.clip(output, 0, 1)
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_xml', required=True)
    parser.add_argument('--model_bin', required=True)
    parser.add_argument('--input_size', type=int, nargs=2, default=[256,256])
    args = parser.parse_args()

    core = Core()
    model = core.read_model(args.model_xml, args.model_bin)
    compiled_model = core.compile_model(model=model, device_name="CPU")

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = natsorted(glob(os.path.join(args.input_dir, '*.*')))

    input_layer = compiled_model.input(0)

    for path in image_paths:
        img = preprocess_image(path, tuple(args.input_size))
        result = compiled_model([img])[compiled_model.output(0)]
        output_img = postprocess_output(result)
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(args.output_dir, filename), output_img)
        print(f"Processed and saved {filename}")

if __name__ == "__main__":
    main()