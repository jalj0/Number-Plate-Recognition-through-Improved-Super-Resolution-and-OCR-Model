import cv2
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

folder_path = '/home1/jalaj_l/Proposed/Rodosol-ALPR-SR/HR'

image_data = []

files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

for filename in tqdm(files, desc="Processing Images"):
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)

    if img is not None:
        height, width = img.shape[:2]
        aspect_ratio = width / height

        image_data.append([filename, width, height, aspect_ratio])

df = pd.DataFrame(image_data, columns=['Image Name', 'Width', 'Height', 'Aspect Ratio'])

output_csv_path = 'HR_data.csv'
df.to_csv(output_csv_path, index=False)

print(f"Image details saved to {output_csv_path}")

##############################################################################

csv_file_path = 'HR_data.csv'
df = pd.read_csv(csv_file_path)

max_height = df['Height'].max()
min_height = df['Height'].min()

max_width = df['Width'].max()
min_width = df['Width'].min()

max_ar = df['Aspect Ratio'].max()
min_ar = df['Aspect Ratio'].min()

avg_height = df['Height'].mean()
avg_width = df['Width'].mean()
avg_aspect_ratio = df['Aspect Ratio'].mean()

print(f"Max Height: {max_height}")
print(f"Min Height: {min_height}")
print(f"Max Width: {max_width}")
print(f"Min Width: {min_width}")
print(f"Max Aspect Ratio: {max_ar}")
print(f"Min Aspect Ratio: {min_ar}")
print(f"Average Height: {avg_height:.2f}")
print(f"Average Width: {avg_width:.2f}")
print(f"Average Aspect Ratio: {avg_aspect_ratio:.2f}")

########################################################################

# distribution of Height
plt.figure(figsize=(8, 6))
plt.hist(df['Height'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Image Height')
plt.xlabel('Height (pixels)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('height_distribution.png')
plt.close()

# Width
plt.figure(figsize=(8, 6))
plt.hist(df['Width'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Image Width')
plt.xlabel('Width (pixels)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('width_distribution.png')
plt.close()

# Aspect Ratio
plt.figure(figsize=(8, 6))
plt.hist(df['Aspect Ratio'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of Image Aspect Ratio')
plt.xlabel('Aspect Ratio (Width/Height)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('aspect_ratio_distribution.png')
plt.close()

print("Plots saved as 'height_distribution.png', 'width_distribution.png', and 'aspect_ratio_distribution.png'")

# Plot scatter plot between Width and Height
plt.figure(figsize=(8, 6))
plt.scatter(df['Width'], df['Height'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Image Width vs. Height')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True)
plt.savefig('width_vs_height_scatter.png')
plt.close()