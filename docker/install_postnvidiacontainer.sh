#!/bin/bash
# Install Docker on Ubuntu/Debian
sudo apt-get update
#sudo apt-get install -y docker.io
# Start and enable Docker service
#sudo systemctl start docker
#sudo systemctl enable docker
#sudo groupadd docker
# Check if the 'docker' group exists
if grep -q '^docker:' /etc/group; then
    echo "The 'docker' group already exists."
else
    # Create the 'docker' group
    sudo groupadd docker
    echo "The 'docker' group has been created."
fi
addusercommand="sudo usermod -aG docker $USER"
#sudo usermod -aG docker $USER
# Execute the commands
eval "$addusercommand" || true  # Execute command1 and ignore errors

#log out and log back again, or run the following command
newgrp docker
echo "Finished docker rootless"
echo "Start installing NVIDIA Container toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
echo "Finished"