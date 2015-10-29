from __future__ import absolute_import, division, print_function

import tempfile
import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

from .. import data
from .. import get_hashes


def test_check_hashes():
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(b'Some data')
        temp.flush()
        fname = temp.name
        d = {fname: "5b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert data.check_hashes(d)
        d = {fname: "4b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert not data.check_hashes(d)

def test_get_hashes(): 
    url = 'http://www.jarrodmillman.com/rcsds/_downloads/ds107_sub001_highres.nii'
    file_name = 'ds107_sub001_highres.nii'
    ds107 = urlopen(url)
    output = open(file_name,'wb')
    output.write(ds107.read())
    output.close()
    file_hashes = get_hashes.get_all_hashes('.')
    assert(file_hashes['./ds107_sub001_highres.nii'] == 'fd733636ae8abe8f0ffbfadedd23896c')
    os.remove('ds107_sub001_highres.nii')
