#!/bin/bash
# Install Docker on Ubuntu/Debian
sudo apt-get update
#sudo apt-get install -y docker.io
# Start and enable Docker service
#sudo systemctl start docker
#sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
#log out and log back again, or run the following command
newgrp docker
