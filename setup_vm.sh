#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# The Git repository containing your Dockerfile and application code.
GIT_REPO_URL="https://github.com/VaibhavPawar212303/architechture.git"
# A safe, standard directory for web applications.
# Cloning into the root directory ("/") is strongly discouraged.
APP_DIR="/opt/phi3-vision-api"

# --- Logging ---
# Redirect all output to a log file and the system's logger for easy debugging.
exec > >(tee /var/log/startup-script.log | logger -t startup-script -s 2>/dev/console) 2>&1

echo "--- Starting VM Setup ---"

# --- Install Dependencies: Git & Docker ---
echo "Updating package lists and installing dependencies..."
# Use sudo for system commands and noninteractive for automated installs.
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y git docker.io

echo "Starting and enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# --- Deploy Application ---
echo "Cloning application from ${GIT_REPO_URL} into ${APP_DIR}..."
# Create the directory if it doesn't exist and then clone into it.
sudo mkdir -p "${APP_DIR}"
sudo git clone "${GIT_REPO_URL}" "${APP_DIR}"
cd "${APP_DIR}"

# --- Build and Run Docker Container ---
echo "Building Docker image..."
# Run Docker commands with sudo for proper permissions.
sudo docker build -t phi3-vision-api .

echo "Running Docker container..."
sudo docker run -d -p 8000:8000 --name phi3-container --restart=always phi3-vision-api

echo "--- ✅ VM Setup Complete ---"