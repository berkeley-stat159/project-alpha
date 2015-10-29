from __future__ import absolute_import, division, print_function

import tempfile
import os

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
    file_hashes = get_hashes.get_all_hashes('.')
    assert(file_hashes['./__init__.py'] == 'd41d8cd98f00b204e9800998ecf8427e')
