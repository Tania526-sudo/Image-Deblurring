import csv
import os

def log_results(csv_path, model_name, dataset_name, psnr, ssim):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Dataset', 'PSNR', 'SSIM'])
        writer.writerow([model_name, dataset_name, f"{psnr:.2f}", f"{ssim:.4f}"])
    print(f"Results logged in {csv_path}")

if __name__ == "__main__":
    
    log_results('results_log.csv', 'MPRNet', 'GoPro', 32.56, 0.965)