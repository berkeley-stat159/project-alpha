import json
from get_all_hashes import get_all_hashes

# Get hashes for all files in all subdirectories of the 
# decompressed ds009 directory. 
if __name__ == "__main__":
    file_hashes = get_all_hashes('ds009')
    with open('ds009_hashes2.json', 'w') as out:
        json.dump(file_hashes, out)
