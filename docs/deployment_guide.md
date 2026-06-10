# VitaVoice Production Deployment Guide

This document describes how to deploy the VitaVoice platform in a production environment using Docker Compose and Nginx.

---

## 1. Single-Command Docker Compose Deployment

The entire system (frontend + backend + models) is orchestrated via Docker Compose.

### Prerequisites
- Install **Docker Engine** (v20.10+) and **Docker Compose** (v2.0+).
- Ensure ports `8000` (backend) and `3000` (frontend) are free on the host.

### Build and Run
From the root directory of the project, run:
```bash
docker-compose up --build -d
```
- `-d` runs the containers in detached (background) mode.
- `--build` forces Docker to rebuild images (baking the Wav2Vec 2.0 checkpoints and frontend assets in).

---

## 2. Production Reverse Proxy Configuration (Nginx)

In a production environment, you should never expose the raw Node or FastAPI ports directly. Use an Nginx reverse proxy to handle SSL termination, static file caching, and routing.

Below is a recommended Nginx configuration (`/etc/nginx/sites-available/vitavoice`):

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

    # Frontend Assets (React SPA Container)
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API Endpoints (FastAPI Container)
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

## 3. Storage and Backup Volumes

The `docker-compose.yml` mounts three host directories into the backend:
1. `./backend/uploads`: Ephemeral audio upload workspace.
2. `./backend/reports`: Ephemeral clinical screening reports (PDFs).
3. `./ml/checkpoints`: Machine learning model weights (scaler, classifier, UMAP/PCA reducers).

### Security Recommendations
- **Upload Pruning**: Ensure the async file pruning job in FastAPI is active (it is enabled by default on startup, deleting all reports and audio files older than 1 hour).
- **Directory Permissions**: Grant write permissions only to the user running the Docker daemon:
  ```bash
  chmod -R 700 ./backend/uploads ./backend/reports
  ```
