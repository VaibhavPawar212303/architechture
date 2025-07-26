#!/bin/bash
# Remove set -e temporarily to better debug issues
# set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# The Git repository containing your Dockerfile and application code.
GIT_REPO_URL="https://github.com/VaibhavPawar212303/architechture.git"
# A safe, standard directory for web applications.
# Cloning into the root directory ("/") is strongly discouraged.
APP_DIR="/opt/phi3-vision-api"

# --- Logging ---
# Redirect all output to a log file and the system's logger for easy debugging.
exec > >(tee /var/log/startup-script.log | logger -t startup-script -s 2>/dev/console) 2>&1

# Function to handle errors
handle_error() {
    local exit_code=$?
    local line_number=$1
    echo "ERROR: Script failed at line $line_number with exit code $exit_code"
    echo "Current working directory: $(pwd)"
    echo "Current user: $(whoami)"
    echo "Environment variables:"
    env | grep -E "(USER|HOME|PATH)" || true
    exit $exit_code
}

# Set up error handling
trap 'handle_error $LINENO' ERR

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
echo "Current user: $(whoami)"
echo "Current working directory: $(pwd)"

# Remove existing directory if it exists
if [ -d "${APP_DIR}" ]; then
    echo "Removing existing directory ${APP_DIR}..."
    rm -rf "${APP_DIR}" || {
        echo "Failed to remove existing directory, trying with sudo..."
        sudo rm -rf "${APP_DIR}"
    }
fi

# Create the directory
echo "Creating directory ${APP_DIR}..."
mkdir -p "${APP_DIR}" || {
    echo "Failed to create directory, trying with sudo..."
    sudo mkdir -p "${APP_DIR}"
}

# Clone the repository
echo "Cloning repository..."
if git clone "${GIT_REPO_URL}" "${APP_DIR}"; then
    echo "✅ Git clone successful"
else
    echo "❌ Git clone failed, trying with sudo..."
    if sudo git clone "${GIT_REPO_URL}" "${APP_DIR}"; then
        echo "✅ Git clone with sudo successful"
    else
        echo "❌ Git clone failed completely"
        echo "Checking git installation..."
        which git
        git --version
        echo "Checking network connectivity..."
        curl -I https://github.com || true
        echo "Trying to clone to /tmp first..."
        git clone "${GIT_REPO_URL}" /tmp/phi3-test && {
            echo "Clone to /tmp successful, moving to ${APP_DIR}..."
            sudo cp -r /tmp/phi3-test/* "${APP_DIR}/"
            rm -rf /tmp/phi3-test
        }
    fi
fi

# Verify the clone worked
if [ ! -f "${APP_DIR}/main.py" ] && [ ! -f "${APP_DIR}/Dockerfile" ]; then
    echo "❌ Repository structure check failed"
    echo "Contents of ${APP_DIR}:"
    ls -la "${APP_DIR}" || true
    
    # Try to find Python files
    echo "Looking for Python files in ${APP_DIR}:"
    find "${APP_DIR}" -name "*.py" -o -name "Dockerfile" -o -name "requirements.txt" 2>/dev/null || true
    
    echo "❌ CRITICAL: Required files not found. Exiting..."
    exit 1
fi

cd "${APP_DIR}"
echo "✅ Successfully changed to ${APP_DIR}"
echo "Current directory contents:"
ls -la

# Set proper permissions
echo "Setting proper permissions..."
chown -R $USER:$USER "${APP_DIR}" || sudo chown -R $USER:$USER "${APP_DIR}"

# --- Build and Run Docker Container ---
echo "Building Docker image..."

# Verify we have the necessary files
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files in current directory:"
    ls -la
    
    # Look for Dockerfile in subdirectories
    echo "Searching for Dockerfile in subdirectories..."
    find . -name "Dockerfile" -type f 2>/dev/null || true
    
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "⚠️ requirements.txt not found, creating one..."
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
torch==2.1.1
torchvision==0.16.1
transformers==4.36.0
accelerate==0.25.0
Pillow==10.1.0
requests==2.31.0
numpy==1.24.4
packaging==23.2
EOF
fi

if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    echo "Searching for Python files..."
    find . -name "*.py" -type f 2>/dev/null || true
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a | grep -q phi3-container; then
    echo "Stopping and removing existing container..."
    docker stop phi3-container 2>/dev/null || true
    docker rm phi3-container 2>/dev/null || true
fi

# Remove existing image if it exists
if docker images | grep -q phi3-vision-api; then
    echo "Removing existing image..."
    docker rmi phi3-vision-api 2>/dev/null || true
fi

# Build new image
echo "Building Docker image (this may take several minutes)..."
if docker build -t phi3-vision-api .; then
    echo "✅ Docker build successful"
else
    echo "❌ Docker build failed"
    echo "Checking Dockerfile contents:"
    cat Dockerfile
    exit 1
fi

echo "Running Docker container..."
if docker run -d \
    -p 8000:8000 \
    --name phi3-container \
    --restart=always \
    -e MODEL_ID="microsoft/Phi-3-vision-128k-instruct" \
    -e MAX_NEW_TOKENS="500" \
    -e PORT="8000" \
    phi3-vision-api; then
    echo "✅ Docker container started successfully"
else
    echo "❌ Failed to start Docker container"
    exit 1
fi

# Wait for container to start
echo "Waiting for container to start..."
sleep 30

# Verify container is running
if ! docker ps | grep -q phi3-container; then
    echo "❌ ERROR: Container failed to start"
    echo "Container logs:"
    docker logs phi3-container 2>/dev/null || echo "Could not retrieve container logs"
    echo "Docker system info:"
    docker info || true
    exit 1
else
    echo "✅ Container is running"
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
    docker logs phi3-container --tail 50 2>/dev/null || echo "Could not retrieve container logs"
    echo "Container status:"
    docker ps -a | grep phi3-container || true
    echo "Port check:"
    netstat -tlnp | grep :8000 || ss -tlnp | grep :8000 || echo "Port 8000 not listening"
fi

# Display container status
echo "Container status:"
docker ps | grep phi3-container || echo "Container not found in running containers"

# Re-enable exit on error for final commands
set -e

echo "--- ✅ VM Setup Complete ---"
echo "API should be available at: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):8000"
echo "Health endpoint: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):8000/health"