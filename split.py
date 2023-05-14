import csv
import os
from PIL import Image
import shutil
# Define the input file path and the directory containing the fundus images
input_file = "../ODIR-5K/ODIR-5K/input.csv"
image_dir = "../ODIR-5K/ODIR-5K/Training Images"

dir_1 = "1"
dir_0 = "0"

# Define a list to store the names of the fundus images with age related macular degeneration
amd_images = []
simple = []
c=0
d=0

# Read the input CSV file
with open(input_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Check if amd is present
        if row["A"] == "1":
            # Check the left and right diagnostic keywords for age related macular degeneration
            left_keywords = row["Left-Diagnostic Keywords"]
            right_keywords = row["Right-Diagnostic Keywords"]
            if "Macular Degeneration".lower() in left_keywords:
                # Add the corresponding left and/or right fundus image names to the list
                amd_images.append(row["Left-Fundus"])
            else:
                simple.append(row["Left-Fundus"])
            if "Macular Degeneration".lower() in right_keywords:
                amd_images.append(row["Right-Fundus"])
            else:
                simple.append(row["Right-Fundus"])
        else:
            simple.append(row["Left-Fundus"])
            simple.append(row["Right-Fundus"])
print(os.path.join(image_dir, row["Left-Fundus"]))
# Print the list of fundus image names with age related macular degeneration
print(amd_images)
print(len(amd_images))
print(len(simple))
print(len(amd_images)+len(simple))
print(os.path.join(image_dir,row["Left-Fundus"]).replace("\\","/"))

print(os.path.join(image_dir,amd_images[0]).replace("\\","/"))

for image_name in amd_images:
    src_path = os.path.join(image_dir,image_name).replace("\\","/")
    dst_path = dir_1
    shutil.copy(src_path,dst_path)

for image_name in simple:
    src_path = os.path.join(image_dir,image_name).replace("\\","/")
    dst_path = dir_0
    shutil.copy(src_path,dst_path)

print(len(os.listdir(dir_1)))
print(len(os.listdir(dir_0)))
print(len(os.listdir(dir_1))+len(os.listdir(dir_0)))