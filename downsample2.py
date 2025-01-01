import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def gauss_noise(img, kernel_size=(5, 5)):
    img_np = np.array(img, copy=True)
    img_np = cv2.GaussianBlur(img_np, kernel_size, 0)
    return Image.fromarray(img_np)

def downsample_image(hr_image, lr_image):
    hr_image_np = np.array(hr_image)

    lr_width, lr_height = lr_image.size

    hr_downsampled_np = cv2.resize(hr_image_np, (lr_width, lr_height), interpolation=cv2.INTER_AREA)

    hr_downsampled_image = Image.fromarray(hr_downsampled_np)

    # image = gauss_noise(hr_downsampled_image)
    
    return hr_downsampled_image

def process_images(input_folder, output_folder, lr_image_path):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the LR image
    lr_image = Image.open(lr_image_path).convert('RGB')

    # Get list of all .jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

    # Process each image with a progress bar
    for filename in tqdm(image_files, desc="Processing Images"):
        hr_image_path = os.path.join(input_folder, filename)
        hr_image = Image.open(hr_image_path).convert('RGB')
        
        # Downsample the image
        downsampled_image = downsample_image(hr_image, lr_image)
        
        # Save the downsampled image to the output folder
        output_image_path = os.path.join(output_folder, filename)
        downsampled_image.save(output_image_path)

    print(f"Processed images saved to {output_folder}")

# Example usage
input_folder = '/home1/jalaj_l/Proposed/Rodosol-ALPR-SR/HR'
output_folder = '/home1/jalaj_l/Proposed/hr_downsampled'
lr_image_path = '/home1/jalaj_l/plate_extracted.jpg'
process_images(input_folder, output_folder, lr_image_path)
