import torch
import argparse
import subprocess

def convert_to_onnx(model, dummy_input_shape, onnx_path, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    dummy_input = torch.randn(*dummy_input_shape)
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11)
    print(f"ONNX model saved to {onnx_path}")

def convert_to_ir(onnx_path, output_dir, input_shape):
    mo_command = f"mo --input_model {onnx_path} --output_dir {output_dir} --input_shape {input_shape}"
    print(f"Running Model Optimizer:\n{mo_command}")
    subprocess.run(mo_command, shell=True, check=True)
    print(f"IR model saved to {output_dir}")

if __name__ == "__main__":
    import runpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to PyTorch weights')
    parser.add_argument('--output_dir', required=True, help='Output directory for IR model')
    args = parser.parse_args()

    model_module = runpy.run_path('MPRNet.py')
    model = model_module['MPRNet']()

    onnx_path = args.output_dir + '/model.onnx'
    dummy_input_shape = (1, 3, 256, 256)  # Adjust based on model input

    convert_to_onnx(model, dummy_input_shape, onnx_path, args.weights)
    convert_to_ir(onnx_path, args.output_dir, str(list(dummy_input_shape)))