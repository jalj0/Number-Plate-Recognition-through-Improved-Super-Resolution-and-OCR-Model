import cv2
import numpy as np
from PIL import Image

def gauss_noise(img, kernel_size=(3, 3)):
    img_np = np.array(img, copy=True)
    img_np = cv2.GaussianBlur(img_np, kernel_size, 0)
    return Image.fromarray(img_np)
    
def add_random_noise(img, mean=0, std=1):
    img_np = np.array(img, copy=True)
    noise = np.random.normal(mean, std, img_np.shape).astype(np.uint8)
    noisy_img_np = cv2.add(img_np, noise)
    return Image.fromarray(np.clip(noisy_img_np, 0, 255).astype(np.uint8))

def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    img_np = np.array(img, copy=True)
    img_np = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
    return Image.fromarray(img_np)

def downsample_image(hr_image_path, lr_image_path):
    hr_image = Image.open(hr_image_path).convert('RGB')
    hr_image_np = np.array(hr_image)

    lr_image = Image.open(lr_image_path).convert('RGB')
    lr_width, lr_height = lr_image.size

    # Downsample HR image to match the LR image's resolution
    hr_downsampled_np = cv2.resize(hr_image_np, (lr_width, lr_height), interpolation=cv2.INTER_AREA)
    hr_downsampled_image = Image.fromarray(hr_downsampled_np)

    hr_noisy_image = add_random_noise(hr_downsampled_image)
    hr_filtered_image = gauss_noise(hr_noisy_image)

    return hr_filtered_image

hr_image_path = '/<path>/HR/img_000001.jpg'
lr_image_path = '/<path>/plate_extracted.jpg'
downsampled_image = downsample_image(hr_image_path, lr_image_path)
downsampled_image.save('downsampled_r_b.jpg')
