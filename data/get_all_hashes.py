import os
import sys

from . import get_hashes

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
            file_hashes[file_path] = get_hashes.generate_file_md5(file_path)
    return file_hashes 
