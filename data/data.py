from __future__ import print_function, division

import json
from get_hashes import check_hashes

# Check the ds009 data. 
if __name__ == "__main__":
    with open('ds009_hashes.json', 'r') as fp:
        d = json.load(fp)
    check_hashes(d)
