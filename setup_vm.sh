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
echo "Timestamp: $(date)"

# --- Install Dependencies: Git & Docker ---
echo "Updating package lists and installing dependencies..."
# Use sudo for system commands and noninteractive for automated installs.
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y git docker.io curl

echo "Starting and enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (for future SSH sessions)
sudo usermod -aG docker $USER

echo "Verifying Docker installation..."
if ! sudo docker --version; then
    echo "ERROR: Docker installation failed"
    exit 1
fi

# --- Deploy Application ---
echo "Cloning application from ${GIT_REPO_URL} into ${APP_DIR}..."
# Remove existing directory if it exists
if [ -d "${APP_DIR}" ]; then
    sudo rm -rf "${APP_DIR}"
fi

# Create the directory if it doesn't exist and then clone into it.
sudo mkdir -p "${APP_DIR}"
sudo git clone "${GIT_REPO_URL}" "${APP_DIR}"
cd "${APP_DIR}"

# Set proper permissions
sudo chown -R $USER:$USER "${APP_DIR}"

# --- Build and Run Docker Container ---
echo "Building Docker image..."
# Stop and remove existing container if it exists
if sudo docker ps -a | grep -q phi3-container; then
    echo "Stopping and removing existing container..."
    sudo docker stop phi3-container || true
    sudo docker rm phi3-container || true
fi

# Remove existing image if it exists
if sudo docker images | grep -q phi3-vision-api; then
    echo "Removing existing image..."
    sudo docker rmi phi3-vision-api || true
fi

# Build new image
sudo docker build -t phi3-vision-api .

echo "Running Docker container..."
sudo docker run -d \
    -p 8000:8000 \
    --name phi3-container \
    --restart=always \
    -e MODEL_ID="microsoft/Phi-3-vision-128k-instruct" \
    -e MAX_NEW_TOKENS="500" \
    -e PORT="8000" \
    phi3-vision-api

# Wait for container to start
echo "Waiting for container to start..."
sleep 30

# Verify container is running
if ! sudo docker ps | grep -q phi3-container; then
    echo "ERROR: Container failed to start"
    echo "Container logs:"
    sudo docker logs phi3-container
    exit 1
fi

# Test the health endpoint
echo "Testing API health endpoint..."
max_retries=10
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API health check passed"
        break
    else
        echo "API not ready yet, retrying in 30 seconds... ($((retry_count + 1))/$max_retries)"
        sleep 30
        retry_count=$((retry_count + 1))
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "⚠️  WARNING: API health check failed after $max_retries attempts"
    echo "Container logs:"
    sudo docker logs phi3-container --tail 50
fi

# Display container status
echo "Container status:"
sudo docker ps | grep phi3-container

echo "--- ✅ VM Setup Complete ---"
echo "API should be available at: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):8000"
echo "Health endpoint: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):8000/health"