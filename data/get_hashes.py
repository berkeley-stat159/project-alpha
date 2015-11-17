from __future__ import print_function, division

import hashlib

# Get md5 hashes for `filename`. 
def generate_file_md5(filename, blocksize=2**20):
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()

# `d` is a dictionary in which the keys are file names and the items
# are the hashes corresponding to the keys.
# Checks whether the hash of each file key in `d` matches the 
# corresponding hash item. 
def check_hashes(d):
    all_good = True
    for k, v in d.items():
        digest = generate_file_md5(k)
        if v == digest:
            print("The file {0} has the correct hash.".format(k))
        else:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
    return all_good
