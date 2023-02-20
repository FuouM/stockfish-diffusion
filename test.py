import glob

root_dir = "F:\\1 New Home\\Archive 8\\stockfish-diffusion\\dataset\\train"

import os

def list_files(path):
    """
    List all files recursively under the given path
    """
    file_list = []
    for root, directories, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_list.append(file_path)            
    return file_list

print(list_files(root_dir))
# print(os.path.isdir(root_dir))
# print("Hello world")

# print(os.listdir(root_dir))