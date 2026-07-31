"""Test suite for FlatRun.

Run with::

    cd src && python -m pytest flatrun/tests -q

The tests build a tiny synthetic SafeTensors checkpoint in a tmpdir
and exercise the public API end-to-end.
"""