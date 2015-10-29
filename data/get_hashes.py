import os
import json
from data import generate_file_md5

def get_hashes(data_dir, output_file):
    """
    Gets all files in all subdirectories of the supplied directory. 
    Creates a dictionary where the paths to the files are the keys 
    and the values are their corresponding hashes. 
    
    Returns nothing, but dumps the dictionary to a JSON file.
    
    Parameters
    ----------
    data_dir: path to a directory
        The directory to search for files.
    output_file: JSON file name
        The json file to dump the resulting dictionary.
    """
    file_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hashes[file_path] = generate_file_md5(file_path)
    with open(output_file, 'w') as out:
        json.dump(file_hashes, out)

if __name__ == "__main__":
    get_hashes('ds009', 'ds009_hashes.json')
