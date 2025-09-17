# Install packages
!pip install pillow torch torchvision
# download pretrained model requirement
!git clone https://github.com/ultralytics/yolov5.git

# package import
import torch
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
import shutil
import os
from torchvision import transforms
from PIL import Image, ImageEnhance
import cv2

os.chdir("/content/yolov5")
#Install required package For YOLO model
!pip install -r requirements.txt



os.makedirs("Project/images/train")
os.makedirs("Project/labels/train")

source_folder = "/content/drive/MyDrive/Project/NumberPlate Datset with annotation/yolo_labels/"

destination_folder = os.path.join(os.getcwd(), "Project/labels/train/")

# Loop through all files in the source folder move label files
for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)

    # Check if it's a file(not folder)
    if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)
        print(f"copied: {source_file} -> {destination_file}")


destination_folder1 = os.path.join(os.getcwd(), "Project/images/train/")
source_folder1 = "/content/drive/MyDrive/Project/NumberPlate Datset with annotation/images/images/"

# move the image file to train folder for those having annotation data
for filename in os.listdir(source_folder):
  Annfile = str( filename.replace(".txt","") )
  for imagefile in os.listdir(source_folder1):
    if Annfile in imagefile:
      source_file = os.path.join(source_folder1, imagefile)
      destination_file = os.path.join(destination_folder1, imagefile)
      # Check if it's a file(not folder)
      if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)
        print(f"copied: {source_file} -> {destination_file}")


#Check image is blur or not
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold  # True = blurry

# Image enhancement
folder_path = os.path.join(os.getcwd(), "Project/images/train/")
for file_name in os.listdir(folder_path):
  file_path = os.path.join(folder_path, file_name)
  image = cv2.imread(file_path)
  if not is_blurry(image, 100):
    print("Blurry Image")
    print(f"Image at : {file_path}")
    im=Image.open(file_path)
    enhancer = ImageEnhance.Sharpness(im)
    EnhancedImage = enhancer.enhance(3.0)
    print(f"Enhanced Image: {EnhancedImage}")
    #Save back with the same name and format
    folder = os.path.dirname(file_path)           # get the folder path
    filename = os.path.basename(file_path)        # get the filename
    save_path = os.path.join(folder, filename)
    EnhancedImage.save(save_path)
    print(f"Image saved at {save_path}")
  else:
    print("Not Blurry Image")

# Preprocessing for saving as image
transform_for_save = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])
# For saving, transform tensor -> image
to_pil = transforms.ToPILImage()


#Load, Preprocess, and Save the Image Back
image_dir = os.path.join(os.getcwd(), "Project/images/train/")
for filename in os.listdir(image_dir):
  img_path = os.path.join(image_dir, filename)
  # Skip directories like '.ipynb_checkpoints'
  if os.path.isdir(img_path):
    continue
  # Open image
  img = Image.open(img_path)
  # Apply preprocessing (resize, crop)
  processed = transform_for_save(img)
  # Save back with the same name and format
  save_path = os.path.join(image_dir, filename)
  processed.save(save_path)
  print(f"Saved: {save_path}")


# Setup parameter - only for Yolo
yolo_parameters = {

                   "train": os.path.join(os.getcwd(), "Project/images/train/"),
                   "val": os.path.join(os.getcwd(), "Project/images/train/"),
                   "nc":1,
                   "names": ["Plate"]
}

with open("Project/train.yaml", "w") as f:
  yaml.dump(yolo_parameters, f)

# Training model
!python train.py --img 500 --batch 5 --epochs 50 --data /content/yolov5/Project/train.yaml --weights yolov5s.pt

!cp /content/yolov5/runs/train/exp/weights/best.pt /content/drive/MyDrive/Project/NumPlateBest.pt

### Final model saved as NumPlateBest.pt
