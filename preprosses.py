import os
import shutil

# define the input folder path
input_folder = "D:\\pneumonia\\Dataset\\chest_xray"

# define the output folder paths
bacteria_folder = "D:\\pneumonia\\Dataset\\out\\BACKTERIAL"
virus_folder = "D:\pneumonia\Dataset\out\VIRAL"

# create the output folders if they don't exist
os.makedirs(bacteria_folder, exist_ok=True)
os.makedirs(virus_folder, exist_ok=True)

# loop through all the files in the input folder
for filename in os.listdir(input_folder):
    # check if the filename contains "bacteria" or "virus"
    if "bacteria" in filename.lower():
        # move the file to the bacteria folder
        shutil.move(os.path.join(input_folder, filename), bacteria_folder)
    elif "virus" in filename.lower():
        # move the file to the virus folder
        shutil.move(os.path.join(input_folder, filename), virus_folder)