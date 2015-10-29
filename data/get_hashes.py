import os
import sys
import json

#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from . import data

def get_all_hashes(data_dir):
    """
    Gets all files in all subdirectories of the supplied directory. 
    Creates a dictionary where the paths to the files are the keys 
    and the values are their corresponding hashes. 
    
    Returns file_hash, a dictionary of hashes. 
    
    Parameters
    ----------
    data_dir: path to a directory
        The directory to search for files.
    """
    file_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hashes[file_path] = data.generate_file_md5(file_path)
    return file_hashes 


if __name__ == "__main__":
    file_hashes = get_all_hashes('ds009')
    with open('ds009_hashes.json', 'w') as out:
        json.dump(file_hashes, out)
