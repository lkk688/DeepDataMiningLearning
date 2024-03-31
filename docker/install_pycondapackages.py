#!/usr/bin/env python

import subprocess

def install_packages(packages):
    """
    Install Python packages using pip.

    Args:
        packages (list): List of package names to install.
    """
    for package in packages:
        try:
            subprocess.check_call(["pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")

def install_conda_packages(packages, channels=None):
    """
    Install Python packages using conda.

    Args:
        packages (list): List of package names to install.
        channels (list, optional): List of additional conda channels. Defaults to None.
    """
    conda_cmd = ["conda", "install", "-y"]
    if channels:
        for channel in channels:
            conda_cmd.extend(["-c", channel])

    for package in packages:
        try:
            subprocess.check_call(conda_cmd + [package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error installing {package}")

if __name__ == "__main__":
    # List of packages to install
    condapackages_to_install = [
        "numpy",
        "pandas",
        "matplotlib",
        "requests",
        # Add more packages here
    ]

    additional_channels = [
        "conda-forge",
        # Add more channels here
    ]

    pippackages_to_install = [
        "numpy",
        "pandas",
        "matplotlib",
        "requests",
        # Add more packages here
    ]

    install_conda_packages(packages_to_install, channels=additional_channels)

    install_packages(pippackages_to_install)

#python install_conda_packages.py