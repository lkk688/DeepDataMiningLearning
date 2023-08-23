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

Install Python on Windows
--------------------------

There are three methods of installing python you can choose from on Windows

Using The Microsoft Store
~~~~~~~~~~~~~~~~~~~~~~~~~~

Microsoft hosts a community release of Python 3 in the Microsoft Store. This is the recommended way to install Python on Windows because it handles updates automatically and can be uninstalled easily too.

To use this method: 1) open the Microsoft store and search for Python; 2)Pick the newest version and install it

With the official installer
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can download a Python installer from the official Python download website (https://www.python.org/downloads/). This method does not give you automatic updates. When you use this installer, make sure you mark the checkbox that says "Add Python to PATH".
    * You can also check the system environment variable, you can see the path of "C:\Program Files\Python39\" and "C:\Program Files\Python39\Scripts\" are included

.. code-block:: console

    PS C:\Users\lkk68> python -V
    Python 3.9.10
    PS C:\Users\lkk68> pip -V
    pip 21.2.4 from C:\Program Files\Python39\lib\site-packages\pip (python 3.9)
    #install and setup virtualenv
    pip install virtualenv
    python -m venv --system-site-packages .\venv
    #activate the virtual environment
    .\venv\Scripts\activate
    #deactivate the virtual environment
    deactivate

When you run "activate", if you see the following error of "running scripts is disabled on this system. For more information, see about_Execution_Policies", you need to setup the execution policy on Windows

.. code-block:: console

    PS C:\Users\lkk68\Documents\Developer> Get-ExecutionPolicy
    Restricted
    PS C:\Users\lkk68\Documents\Developer> Get-ExecutionPolicy –List
    PS C:\Users\lkk68\Documents\Developer> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    PS C:\Users\lkk68\Documents\Developer> Get-ExecutionPolicy
    RemoteSigned
    PS C:\Users\lkk68\Documents\Developer> .\venv\Scripts\activate

With Conda
~~~~~~~~~~~

Use Conda: You can also install python via Anaconda (https://www.anaconda.com/) or Miniconda (https://docs.conda.io/en/latest/miniconda.html). Ref "CondaEnv.rst" for detailed steps.

With WSL2
~~~~~~~~~~~
You can also use Windows Subsystem For Linux (WSL2). To install in WSL, you'll first need to install WSL itself. Check "WSL2.rst" document for detailed steps. After the WSL is installed, you can refer the Linux installation instructions.

Install JupyterLab on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    (.venv) PS C:\Users\lkk\Documents\Developer> pip install jupyterlab
    (.venv) PS C:\Users\lkk\Documents\Developer> pip install ipykernel
    (.venv) PS C:\Users\lkk\Documents\Developer> python -m ipykernel install --user --name=venv
    #start the jupyter lab server
    jupyter lab --ip 0.0.0.0 --no-browser --allow-root


Install Python on Mac
----------------------

On most versions of MacOS before Catalina, a distribution of Python is already included. It is recommended to not use the system python. There are two ways to easily install additional Python 3 for development on a Mac.

Official installer
~~~~~~~~~~~~~~~~~~
You can download an installer from the Python download website: https://www.python.org/downloads/. If you have Apple Silicon, you're going to use the Python download links that include the name: "universal2". If you have an Intel processor, you're going to use the Python download links that include the name: "Intel-only".

After the download completes, open the downloaded pkg file. Run all the defaults for the installer.

Install via HomeBrew
~~~~~~~~~~~~~~~~~~~~
Install python via homebrew: https://brew.sh/. If you have Homebrew installed you can run: "brew search python", then install via:

.. code-block:: console
    brew install python@3.10
    #create a virtual environment
    python -m venv myvenv
    source myvenv/bin/activate
    (myvenv) python -m pip install pip --upgrade
    (myvenv) deactivate

With Conda
~~~~~~~~~~~

Use Conda: You can also install python via Anaconda (https://www.anaconda.com/) or Miniconda (https://docs.conda.io/en/latest/miniconda.html). Ref "MacML.rst" for detailed steps.

.. code-block:: console

  kaikailiu@kaikais-mbp Developer % curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
  kaikailiu@kaikais-mbp Developer % sh Miniconda3-latest-MacOSX-arm64.sh
  #install to /Users/kaikailiu/miniconda3
  (base) kaikailiu@kaikais-mbp ~ % python3 -V #this is system's python
  Python 3.10.10
  (base) kaikailiu@kaikais-mbp ~ % conda --version
  conda 23.3.1
  (base) kaikailiu@kaikais-mbp ~ % conda update conda
  (base) kaikailiu@kaikais-mbp docs % conda create --name mypy310 python=3.10 
  (base) kaikailiu@kaikais-mbp docs % conda activate mypy310
  (mypy310) kaikailiu@kaikais-mbp docs % python3 --version #this is conda's python
  Python 3.10.11
  (mypy310) kaikailiu@kaikais-mbp docs % which python3
  /Users/kaikailiu/miniconda3/envs/mypy310/bin/python3
  (mypy310) kaikailiu@kaikais-mbp docs % which pip3
  /Users/kaikailiu/miniconda3/envs/mypy310/bin/pip3

Python Tutorial (Colab)
-----------------------

https://colab.research.google.com/drive/1KpLTxgvmFzSlmr486zZwfUBUt-U4-ukT?usp=sharing
