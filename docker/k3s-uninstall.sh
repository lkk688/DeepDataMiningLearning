#!/bin/bash

# stop k3s Service
sudo systemctl stop k3s

# Uninstall k3s
sudo /usr/local/bin/k3s-uninstall.sh

# Remove k3s Data and Configuration
sudo rm -rf /var/lib/rancher/k3s

# Remove kubectl Configuration
rm -f ~/.kube/config

echo "k3s has been successfully uninstalled."