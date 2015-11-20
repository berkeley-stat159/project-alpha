""" Tests hash-checking functions. 

Run with:
    nosetests test_data.py
"""

from __future__ import absolute_import, division, print_function

import tempfile
import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

from .. import get_hashes
from .. import get_all_hashes


def test_check_hashes():
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(b'Some data')
        temp.flush()
        fname = temp.name
        d = {fname: "5b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert get_hashes.check_hashes(d)
        d = {fname: "4b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert not get_hashes.check_hashes(d)

def test_get_hashes(): 
    # Download flag text file to current directory. 
    url = 'http://textfiles.com/art/flag'
    file_name = 'flag.txt'
    flag = urlopen(url)
    output = open(file_name,'wb')
    output.write(flag.read())
    output.close()
    # Now get all hashes of all files in current directory.
    file_hashes = get_all_hashes.get_all_hashes('.')
    # The text file should be in the dictionary of hashes, with the 
    # correct hash. 
    assert(file_hashes['./flag.txt'] == 'a0cd62d98374cb5dfbd1c53a220e12fa')
    # Delete the file. 
    os.remove('flag.txt')
