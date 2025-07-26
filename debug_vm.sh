#!/bin/bash

echo "=== VM Debug Script ==="
echo "Timestamp: $(date)"
echo

echo "=== User Information ==="
echo "Current user: $(whoami)"
echo "User ID: $(id)"
echo "Home directory: $HOME"
echo "Current directory: $(pwd)"
echo

echo "=== System Information ==="
echo "OS: $(cat /etc/os-release | head -2)"
echo "Uptime: $(uptime)"
echo "Free space:"
df -h /
echo

echo "=== Network Connectivity ==="
echo "Testing internet connectivity..."
if ping -c 2 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ Internet connectivity OK"
else
    echo "❌ No internet connectivity"
fi

if curl -I https://github.com > /dev/null 2>&1; then
    echo "✅ GitHub accessible"
else
    echo "❌ GitHub not accessible"
fi
echo

echo "=== Git Information ==="
which git
git --version
echo

echo "=== Docker Information ==="
which docker
docker --version
systemctl status docker --no-pager
echo

echo "=== Checking Startup Script ==="
if [ -f "/var/log/startup-script.log" ]; then
    echo "Startup script log exists:"
    tail -20 /var/log/startup-script.log
else
    echo "❌ Startup script log not found"
fi
echo

echo "=== Checking Application Directory ==="
APP_DIR="/opt/phi3-vision-api"
if [ -d "$APP_DIR" ]; then
    echo "✅ Application directory exists"
    echo "Contents:"
    ls -la "$APP_DIR"
    
    if [ -f "$APP_DIR/main.py" ]; then
        echo "✅ main.py found"
    else
        echo "❌ main.py not found"
    fi
    
    if [ -f "$APP_DIR/Dockerfile" ]; then
        echo "✅ Dockerfile found"
    else
        echo "❌ Dockerfile not found"
    fi
else
    echo "❌ Application directory does not exist"
fi
echo

echo "=== Docker Containers ==="
echo "Running containers:"
docker ps
echo
echo "All containers:"
docker ps -a
echo

echo "=== Docker Images ==="
docker images
echo

echo "=== Port Status ==="
echo "Checking port 8000:"
netstat -tlnp | grep :8000 || ss -tlnp | grep :8000 || echo "Port 8000 not listening"
echo

echo "=== Manual Git Clone Test ==="
TEST_DIR="/tmp/git-test"
rm -rf "$TEST_DIR"
echo "Testing git clone to $TEST_DIR..."
if git clone "https://github.com/VaibhavPawar212303/architechture.git" "$TEST_DIR"; then
    echo "✅ Git clone successful"
    echo "Cloned contents:"
    ls -la "$TEST_DIR"
    rm -rf "$TEST_DIR"
else
    echo "❌ Git clone failed"
fi
echo

echo "=== Recent System Logs ==="
echo "Recent startup script logs:"
journalctl -u google-startup-scripts.service --no-pager -n 20 || echo "Could not access startup script logs"

echo
echo "=== Debug Complete ==="