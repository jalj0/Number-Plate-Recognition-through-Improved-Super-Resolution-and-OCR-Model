import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

def process_annotations(folder_path, output_folder, mapping_file_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the mapping file in write mode
    with open(mapping_file_path, "w") as mapping_file:
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith(".xml"):
                xml_path = os.path.join(folder_path, file_name)

                tree = ET.parse(xml_path)
                root = tree.getroot()

                filename = root.find(".//filename").text
                image_path = os.path.join(folder_path, filename)

                try:
                    image = Image.open(image_path)
                except FileNotFoundError:
                    print(f"Image file {filename} not found for XML {file_name}. Skipping...")
                    continue

                # Process each object in the XML
                for obj in root.findall(".//object"):
                    name = obj.find("name").text
                    if len(name) != 10:
                        print(f"{file_name} has length {len(name)}, skipping...")
                        break

                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # Crop the image
                    cropped_image = image.crop((xmin, ymin, xmax, ymax))

                    # Save the cropped image
                    cropped_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
                    cropped_image = cropped_image.convert("RGB")  # Convert to RGB if not already
                    cropped_image.save(cropped_image_path, format="JPEG")

                    # Save the ground truth label 
                    gt_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
                    with open(gt_file_path, "w") as gt_file:
                        gt_file.write(f"plate: {name}\n")
                        gt_file.write("layout: Indian\n")

                    hr_path = f"dataset/HR/{os.path.splitext(filename)[0]}.jpg"
                    lr_path = f"dataset/LR/{os.path.splitext(filename)[0]}.jpg"
                    mapping_file.write(f"{hr_path};{lr_path};testing\n")

    print(f"Cropped images and labels saved to: {output_folder}")
    print(f"Mapping file saved to: {mapping_file_path}")

folder_path = "F:\\Mtech IITKGP\\MTech_Project\\super_res\\lpr\\Proposed\\ivlp_dataset\\test"
output_folder = "F:\\Mtech IITKGP\\MTech_Project\\super_res\\lpr\\Proposed\\ivlp_dataset\\ivlp\\HR"
mapping_file_path = "F:\\Mtech IITKGP\\MTech_Project\\super_res\\lpr\\Proposed\\ivlp_dataset\\ivlp\\split_test.txt"
process_annotations(folder_path, output_folder, mapping_file_path)

