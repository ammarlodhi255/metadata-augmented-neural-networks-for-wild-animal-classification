import os
import shutil

# set the source and destination directories
source_dir = "/media/user-1/CameraTraps/NINA_raw/images/"
dest_dir = '/media/user-1/CameraTraps/NINA_raw/new_images/'

# loop through all files in the source directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # create the path to the source file
        src_path = os.path.join(root, file)
        
        # create the path to the destination file
        dest_path = os.path.join(dest_dir, file)
        
        # copy the file to the destination
        shutil.copy2(src_path, dest_path)

print("Files copied successfully.")