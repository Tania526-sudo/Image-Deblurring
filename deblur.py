import argparse
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

def load_model(model_name='mprnet', device='cpu'):
    if model_name == 'mprnet':
        model = torch.hub.load('swz30/MPRNet', 'MPRNet', pretrained=True)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    model.eval()
    model.to(device)
    return model

def preprocess(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)

def postprocess(tensor):
    img = tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0)) * 255
    return Image.fromarray(img.astype(np.uint8))

def deblur_image(input_path, output_path, model_name='mprnet', device='cpu'):
    model = load_model(model_name, device)
    image = Image.open(input_path).convert('RGB')
    input_tensor = preprocess(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    deblurred_image = postprocess(output)
    deblurred_image.save(output_path)
    print(f"Deblurred image saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image deblurring script")
    parser.add_argument('input', help="Path to blurred image")
    parser.add_argument('output', help="Path to save deblurred image")
    parser.add_argument('--model', default='mprnet', choices=['mprnet'])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    deblur_image(args.input, args.output, args.model, args.device)