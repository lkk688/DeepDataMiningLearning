Python
====================

Python Packages
---------------

The "new standards" refer to a standardized way to specify package metadata (things like package name, author, dependencies) in a pyproject.toml file and the way to build packages from source code using that metadata. You often see this referred to as "pyproject.toml-based builds." If you have created Python packages in the past and remember using a setup.py or a setup.cfg file, the new build standards are replacing that. ref: https://drivendata.co/blog/python-packaging-2023

Flit is a simple way to put Python packages and modules on PyPI: https://flit.pypa.io/en/stable/index.html

To install a package locally for development, run: flit install [--symlink] [--python path/to/python]

.. code-block:: console

    (mypy310) % python3 -m pip install flit
    (mypy310) kaikailiu@kaikais-mbp DeepDataMiningLearning % flit install --symlink

Run this command to upload your code to PyPI: "flit publish"


If you install a package with flit install --symlink, a symlink is made for each file in the external data directory. Otherwise (including development installs with pip install -e), these files are copied to their destination, so changes here won't take effect until you reinstall the package.

