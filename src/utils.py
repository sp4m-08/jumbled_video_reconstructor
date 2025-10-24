import os,shutil

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)

