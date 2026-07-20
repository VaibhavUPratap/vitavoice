# VitaVoice Production Deployment Guide

This document describes how to deploy the VitaVoice platform in a production environment by running the FastAPI backend and serving the compiled Vite React frontend using Nginx.

---

## 1. Backend Production Setup

The FastAPI backend runs on Uvicorn. For production, it is recommended to run it using a process manager like **PM2** or as a **systemd** service to ensure automatic restarts and background execution.

### Installation
1. Initialize the virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Running with PM2
To run the backend in the background and ensure it restarts on system reboot, use PM2:
```bash
# Start backend under PM2
pm2 start "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000" --name "vitavoice-backend"

# Save list of processes for startup
pm2 save
```

---

## 2. Frontend Production Setup

For production, build the Vite React application into static assets and serve them directly via Nginx. Do not run the dev server (`npm run dev`) in production.

### Build static assets
Navigate to the frontend directory and build the production bundle:
```bash
cd frontend
npm install
npm run build
```
This generates a compiled production bundle inside the `frontend/dist/` directory.

---

## 3. Production Reverse Proxy Configuration (Nginx)

Nginx is used to serve the static frontend assets directly and reverse-proxy the API requests to the FastAPI backend running on port 8000.

Below is the recommended Nginx configuration (`/etc/nginx/sites-available/vitavoice`):

```nginx
server {
    listen 80;
    server_name vitavoice.yourdomain.com;
    return 301 https://$host$request_uri; # Redirect all HTTP to HTTPS
}

server {
    listen 443 ssl http2;
    server_name vitavoice.yourdomain.com;

    # SSL Certificates (managed via Let's Encrypt / Certbot)
    ssl_certificate /etc/letsencrypt/live/vitavoice.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vitavoice.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend Assets (Serve compiled static files directly)
    location / {
        root /var/www/vitavoice/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API Endpoints (FastAPI Application)
    location /api/v1/ {
        proxy_pass http://127.0.0.1:8000/api/v1/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Enforce file upload size limits (e.g. max 10MB)
        client_max_body_size 10M;
    }

    # Static Reports Folder (direct nginx serving for high speed)
    location /api/v1/reports/ {
        alias /var/www/vitavoice/backend/reports/;
        expires 1h;
        add_header Cache-Control "public, no-transform";
    }
}
```

---

## 4. Directory Permissions & File Cleanup

To ensure maximum security and proper cleanup of temporary audio files and reports:
1. **Upload & Report directories**: Create these directories inside the `backend` folder:
   ```bash
   mkdir -p backend/uploads backend/reports
   ```
2. **Directory Permissions**: Grant write permissions only to the user running the FastAPI app and Nginx:
   ```bash
   chmod -R 700 backend/uploads backend/reports
   ```
3. **Upload Pruning**: Ensure the async file pruning job in FastAPI is active (it is enabled by default on startup, deleting all reports and audio files older than 1 hour).
