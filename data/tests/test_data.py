from __future__ import absolute_import, division, print_function

import tempfile

from .. import data


def test_check_hashes():
    tf = tempfile.NamedTemporaryFile(delete=False)
    fname = tf.name
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(b'Some data')
        temp.flush()
        fname = temp.name
        d = {fname: "5b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert data.check_hashes(d)
        d = {fname: "4b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert not data.check_hashes(d)
