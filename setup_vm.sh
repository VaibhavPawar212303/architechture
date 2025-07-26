#!/bin/bash
set -e # Exit on any error

# --- Variables ---
# IMPORTANT: Update this with the URL to your Git repository
GIT_REPO_URL="https://github.com/your-username/your-phi3-repo.git"
APP_DIR="/opt/phi3-vision-api"

# --- Log everything to a file ---
exec > >(tee /var/log/startup-script.log | logger -t startup-script -s 2>/dev/console) 2>&1

echo "--- Starting VM Setup ---"

# --- Install Dependencies: Git, Docker ---
echo "Installing Git and Docker..."
apt-get update
apt-get install -y git docker.io

# Start and enable Docker
systemctl start docker
systemctl enable docker

# --- Install NVIDIA Container Toolkit ---
echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# --- Clone Application Repository ---
echo "Cloning application from ${GIT_REPO_URL}..."
git clone "${GIT_REPO_URL}" "${APP_DIR}"
cd "${APP_DIR}"

# --- Build and Run Docker Container ---
echo "Building Docker image..."
docker build -t phi3-vision-api .

echo "Running Docker container..."
# The model will download and load inside the container on first run
docker run -d --gpus all -p 8000:8000 --name phi3-container --restart=always phi3-vision-api

echo "--- VM Setup Complete ---"