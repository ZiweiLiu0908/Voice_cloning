import os
import shutil
from tqdm import tqdm

def organize_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate over each file
    for file_name in tqdm(files):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            # Split the file name by underscore
            parts = file_name.split('_')
            if len(parts) >= 1:
                # Extract the first part of the file name
                folder_name = parts[0]
                
                # Create the subfolder if it doesn't exist
                subfolder_path = os.path.join(folder_path, folder_name)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                
                # Move the file to the corresponding subfolder
                shutil.move(os.path.join(folder_path, file_name), subfolder_path)

if __name__ == "__main__":
    folder_path = "F:/dataset/sr/wav/dataset"  # Specify the folder path here
    organize_files(folder_path)