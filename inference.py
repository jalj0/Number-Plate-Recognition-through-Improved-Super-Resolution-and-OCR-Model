import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from training import load_model
from network import Network
import matplotlib.pyplot as plt

# preprocess the input image
def gauss_noise(img):
    _filter = 3
    imgLR = np.array(img, copy=True)
    imgLR = cv2.GaussianBlur(imgLR, (_filter, _filter), 0)
    return Image.fromarray(imgLR)

def preprocess_image(image_path, upscale_factor=2):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # Ensure dimensions are divisible by the upscale_factor
    new_width = (width // upscale_factor) * upscale_factor
    new_height = (height // upscale_factor) * upscale_factor

    if new_width != width or new_height != height:
        print(f"Resizing image from ({width}, {height}) to ({new_width}, {new_height})")
    image = image.resize((new_width, new_height), Image.BICUBIC)

    # Apply Gaussian noise
    image = gauss_noise(image)

    # Convert image to tensor
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Ensure dimensions are divisible by 2 for PixelUnshuffle
    if image_tensor.size(-1) % 2 != 0:
        new_size = (image_tensor.size(-1) // 2) * 2
        print(f"Resizing tensor from ({image_tensor.size(-1)}) to ({new_size})")
        image_tensor = F.interpolate(image_tensor, size=(image_tensor.size(-2), new_size), mode='bilinear', align_corners=False)

    print(image_tensor.shape)
    return image_tensor


# print("yes")

# postprocess the output tensor and save the image
def save_image(tensor, output_path):
    image = tensor.squeeze(0)
    transform = ToPILImage()
    image = transform(image)
    image.save(output_path)

# Inference function
def inference(model_path, input_image_path, output_image_path, upscale_factor=2):
    # Load the model
    model = Network(3, 3).cuda()
    _, model, _, _, _, _ = load_model(model, model_path)  # Unpack model
    model.eval()
    
    # Preprocess the input image
    lr_image_tensor = preprocess_image(input_image_path, upscale_factor=upscale_factor)
    # print("yes2")
    lr_image_tensor = lr_image_tensor.cuda()  # Move the tensor to GPU
    
    # Perform inference
    with torch.no_grad():
        print("yes3")
        sr_image_tensor = model(lr_image_tensor)
        print("yes4")
    
    # Postprocess and save the output image
    save_image(sr_image_tensor.cpu(), output_image_path)
    print(f"Super-resolved image saved to {output_image_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference mode for Super-Resolution Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to the input LR image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output SR image')

    args = parser.parse_args()

    inference(args.model, args.input, args.output)
