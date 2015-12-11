from __future__ import print_function, division

import hashlib
import sys

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

    # Progress bar
    if (len(d) <= 25):
        toolbar_width = len(d)
    else: 
        toolbar_width = 25
    toolbar_ticks = [x * (len(d) // toolbar_width) for x in range(26)][1:]
    sys.stdout.write("Checking hashes :  ")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    i = 1
    for k, v in d.items():
        digest = generate_file_md5(k)
        if v != digest:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
        # Update the progress bar. 
        if i in toolbar_ticks:
            sys.stdout.write("-")
            sys.stdout.flush()
        i += 1

    sys.stdout.write("\n")
    return all_good




