import os
import json
from data import generate_file_md5

def get_hashes(data_folder, output_file):
    file_hashes = {}
    for root, dirs, files in os.walk(data_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hashes[file_path] = generate_file_md5(file_path)
    with open(output_file, 'w') as out:
        json.dump(file_hashes, out)

if __name__ == "__main__":
    get_hashes('ds009', 'ds009_hashes.json')
