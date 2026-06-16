#!/bin/bash
# ==============================================================================
# AWS EC2 Setup and Deployment Script for LegalMind
# This script automates swap space, packages, systemd, and Nginx reverse proxy.
# Run this on your Ubuntu 22.04 LTS instance.
# ==============================================================================

set -e

echo "=== [1/5] Updating system and installing dependencies ==="
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv nginx git curl

echo "=== [2/5] Configuring 2GB Virtual Memory (Swap) ==="
# Prevents Out-Of-Memory (OOM) errors on 1GB RAM (t2.micro) instances during pip install
if [ ! -f /swapfile ]; then
    echo "Creating 2GB swap file..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap space successfully configured!"
else
    echo "Swap file already exists. Skipping."
fi

echo "=== [3/5] Setting up Virtual Environment & Installing Requirements ==="
# Navigate to the app directory (assumed to be run inside /home/ubuntu/LegalMind)
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
echo "Project directory identified as: $PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [4/5] Configuring systemd Service ==="
# This keeps the FastAPI backend running in the background and restarts it on crashes
SERVICE_FILE="/etc/systemd/system/legalmind.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=LegalMind FastAPI Application Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8080
Restart=always
Environment=PATH=$PROJECT_DIR/venv/bin:/usr/bin:/usr/local/bin

[Install]
WantedBy=multi-user.target
EOF

echo "Starting and enabling legalmind service..."
sudo systemctl daemon-reload
sudo systemctl start legalmind
sudo systemctl enable legalmind

echo "=== [5/5] Configuring Nginx Reverse Proxy ==="
# Directs public HTTP (Port 80) traffic to local FastAPI server (Port 8080)
NGINX_CONF="/etc/nginx/sites-available/default"

sudo bash -c "cat > $NGINX_CONF" <<EOF
server {
    listen 80;
    server_name _; # Change this to your domain name when pointing DNS A-record

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

echo "Testing and restarting Nginx..."
sudo nginx -t
sudo systemctl restart nginx

echo "=============================================================================="
echo " Setup complete! LegalMind is running on port 8080 and proxied via Nginx."
echo " Next educational steps to complete your setup:"
echo " 1. Update the .env file in $PROJECT_DIR with cloud Neo4j, Qdrant & Groq keys."
echo " 2. Restart the app service using: sudo systemctl restart legalmind"
echo " 3. Point your domain's A-record to the EC2 Public IP."
echo " 4. Secure the server with SSL (HTTPS) by running:"
echo "    sudo apt-get install certbot python3-certbot-nginx -y"
echo "    sudo certbot --nginx"
echo "=============================================================================="
