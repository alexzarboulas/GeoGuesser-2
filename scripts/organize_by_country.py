# This script organizes images into country-specific folders based on metadata

import os
import shutil # For moving files from raw images directory to labeled images directory
import csv

RAW_IMAGES_DIR = "../data/raw_images" #Raw images directory (created and populated by fetch_images.py)
LABELED_DIR = "../data/labeled_images" #Labeled images directory (populating is the goal of this script)
METADATA_PATH = "../data/metadata.csv"  #Path to metadata CSV file (created by fetch_images.py)

os.makedirs(LABELED_DIR, exist_ok=True) #Make labeled images directory if it doesn't exist

with open(METADATA_PATH, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f) #Read metadata CSV file

    for row in reader: #Iterate over each row (image) in the metadata

        filename, country = row["filename"], row["country"] #Extract filename and country from metadata

        src_path = os.path.join(RAW_IMAGES_DIR, filename) #Construct source path for the image in raw images directory
        if not os.path.exists(src_path): #Check if the source image exists
            print(f"Source image {src_path} does not exist, skipping {filename}.")
            continue

        dest_folder = os.path.join(LABELED_DIR, country) #Construct destination folder path based on country (if it doesn't exist, it will be created)
        dest_path = os.path.join(dest_folder, filename) #Construct destination path for the image in the labeled images directory

        try:

            os.makedirs(dest_folder, exist_ok=True) #Create destination folder if it doesn't exist

            shutil.move(src_path, dest_path) #Move the image from raw images directory to labeled images directory
            print(f"Moved {filename} to {dest_folder}") #Success message

        except Exception as e:
            print(f"Failed to move {filename}: {e}") #Fail message
            continue