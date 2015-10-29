from __future__ import absolute_import, division, print_function

import tempfile
import os
import urllib

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
    testfile = urllib.URLopener()
    testfile.retrieve('http://www.jarrodmillman.com/rcsds/_downloads/ds107_sub001_highres.nii', 'ds107_sub001_highres.nii')
    file_hashes = get_hashes.get_all_hashes('.')
    assert(file_hashes['./ds107_sub001_highres.nii'] == 'fd733636ae8abe8f0ffbfadedd23896c')
    os.remove('ds107_sub001_highres.nii')
