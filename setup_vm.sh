#!/bin/bash
set -e # Exit on any error

# --- Variables ---
GIT_REPO_URL="https://github.com/VaibhavPawar212303/architechture.git"
APP_DIR="/"

# --- Log everything ---
exec > >(tee /var/log/startup-script.log | logger -t startup-script -s 2>/dev/console) 2>&1

echo "--- Starting VM Setup ---"

# --- Install Dependencies: Git, Docker ---
echo "Installing Git and Docker..."
apt-get update
apt-get install -y git docker.io
systemctl start docker
systemctl enable docker

# --- Clone Application Repository ---
echo "Cloning application from ${GIT_REPO_URL}..."
git clone "${GIT_REPO_URL}" "${APP_DIR}"
cd "${APP_DIR}"

# --- Build and Run Docker Container ---
echo "Building Docker image..."
docker build -t phi3-vision-api .

echo "Running Docker container..."
docker run -d -p 8000:8000 --name phi3-container --restart=always phi3-vision-api

echo "--- VM Setup Complete ---"