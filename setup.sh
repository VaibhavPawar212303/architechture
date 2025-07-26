#!/bin/bash

# Phi-3 FastAPI Docker Setup Script

echo "🚀 Setting up Phi-3 with FastAPI Docker container..."

# Create project directory structure
mkdir -p phi3-fastapi
cd phi3-fastapi

# Create models directory for caching
mkdir -p models

echo "📁 Project structure created"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t phi3-fastapi .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Run the container
echo "🏃 Starting container..."
docker run -d \
    --name phi3-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    phi3-fastapi

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully"
    echo "🌐 API will be available at: http://localhost:8000"
    echo "📚 API docs available at: http://localhost:8000/docs"
    echo ""
    echo "⏳ The model is loading... This may take a few minutes on first run."
    echo "   You can check the status at: http://localhost:8000/health"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs: docker logs -f phi3-api"
    echo "   Stop container: docker stop phi3-api"
    echo "   Remove container: docker rm phi3-api"
    
    # Wait a moment and check if container is still running
    sleep 5
    if docker ps | grep -q phi3-api; then
        echo "✅ Container is running healthy"
    else
        echo "⚠️  Container may have issues. Check logs with: docker logs phi3-api"
    fi
else
    echo "❌ Failed to start container"
    exit 1
fi