#!/bin/bash

# Install k3s
curl -sfL https://get.k3s.io | sh -

#Create .kube directory in user's home directory
mkdir -p ~/.kube

# Retrieve Kubernetes configuration and save it to ~/.kube/config
sudo k3s kubectl config view --raw | tee ~/.kube/config

# Change permissions of ~/.kube/config and /etc/rancher/k3s/k3s.yaml
sudo chmod 644 ~/.kube/config
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

# verify installation commands

# sudo systemctl status k3s
# sudo kubectl get nodes
# sudo kubectl get pods --all-namespaces
# sudo kubectl --kubeconfig=/etc/rancher/k3s/k3s.yaml cluster-info