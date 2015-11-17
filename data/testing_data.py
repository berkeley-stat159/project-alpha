from __future__ import print_function, division

import json

from get_hashes import check_hashes

# Dictionary of hashes for the three ds114 files from class. 
d = {'ds114/ds114_sub009_t2r1.nii': 
'709fcca8d33ddb7d0b7d501210c8f51c',
'ds114/ds114_sub009_t2r1_cond.txt': 
'5cb29aed9c9f330afe1af7e69f8aad18',
'ds114/ds114_sub009_t2r1_conv.txt': 
'7f893bc99714c5d8d018c3e46d0d2664'}

# Check the hashes of the ds114 data. 
if __name__ == "__main__":
    check_hashes(d)
