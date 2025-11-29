# 🚀 Deployment Guide - Parameter Pruning Dashboard

This guide will help you deploy your Streamlit Parameter Pruning Dashboard online.

## 📋 Table of Contents
1. [Streamlit Cloud (Recommended - Easiest)](#streamlit-cloud)
2. [Heroku](#heroku)
3. [AWS EC2](#aws-ec2)
4. [Docker Deployment](#docker-deployment)
5. [Pre-Deployment Checklist](#pre-deployment-checklist)

---

## 🌟 Option 1: Streamlit Cloud (Recommended - FREE & Easiest)

Streamlit Cloud is the easiest way to deploy your app. It's free and takes just a few minutes!

### Prerequisites
- A GitHub account
- Your project pushed to a GitHub repository

### Steps:

#### 1. Prepare Your Repository

Make sure your project structure looks like this:
```
parameter_pruning_project/
├── streamlit_app.py          # Main app file
├── requirements.txt           # Dependencies
├── src/                      # Source code
│   ├── model.py
│   ├── train.py
│   └── ...
├── .streamlit/
│   └── config.toml          # Streamlit config (optional)
└── README.md
```

#### 2. Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Ready for deployment"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

#### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** and authorize with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/YOUR_REPO_NAME`
5. Set **Main file path**: `streamlit_app.py`
6. Click **"Deploy"**

That's it! Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### ⚙️ Streamlit Cloud Configuration

Create `.streamlit/config.toml` in your repository:

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

## 🐳 Option 2: Heroku

### Prerequisites
- Heroku account (free tier available)
- Heroku CLI installed

### Steps:

#### 1. Create `Procfile`
```bash
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### 2. Create `setup.sh`
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

#### 3. Update `requirements.txt`
Make sure it includes all dependencies (already done).

#### 4. Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create a new app
heroku create your-app-name

# Deploy
git push heroku main

# Open your app
heroku open
```

---

## ☁️ Option 3: AWS EC2

### Prerequisites
- AWS account
- EC2 instance running Ubuntu

### Steps:

#### 1. Connect to EC2 Instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

#### 2. Install Dependencies
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install Streamlit
pip3 install streamlit
pip3 install -r requirements.txt
```

#### 3. Configure Firewall
```bash
# Allow port 8501
sudo ufw allow 8501/tcp
```

#### 4. Run Streamlit
```bash
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

#### 5. Access Your App
Open browser: `http://YOUR_EC2_PUBLIC_IP:8501`

#### 6. Run as Service (Optional - for auto-start)
Create `/etc/systemd/system/streamlit.service`:
```ini
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/parameter_pruning_project
ExecStart=/usr/local/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

---

## 🐋 Option 4: Docker Deployment

### 1. Create `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create `.dockerignore`
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
pip-log.txt
pip-delete-this-directory.txt
.git
.gitignore
README.md
.env
.venv
data/
saved/
assets/
uploads/
*.pth
*.pdf
```

### 3. Build and Run
```bash
# Build image
docker build -t parameter-pruning-dashboard .

# Run container
docker run -p 8501:8501 parameter-pruning-dashboard
```

### 4. Deploy to Docker Hub / Cloud
```bash
# Tag image
docker tag parameter-pruning-dashboard YOUR_DOCKERHUB_USERNAME/parameter-pruning-dashboard

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/parameter-pruning-dashboard

# Deploy to any cloud platform that supports Docker (AWS ECS, Google Cloud Run, etc.)
```

---

## ✅ Pre-Deployment Checklist

Before deploying, make sure:

- [ ] **All dependencies are in `requirements.txt`**
- [ ] **No hardcoded paths** - Use relative paths
- [ ] **Environment variables** - Use `.env` file or Streamlit secrets
- [ ] **Large files** - Add to `.gitignore` (data/, saved/, assets/)
- [ ] **Secrets** - Never commit API keys or passwords
- [ ] **Test locally** - App runs without errors
- [ ] **README updated** - Include deployment instructions

### 🔒 Security Considerations

1. **Secrets Management**
   - Use Streamlit secrets for sensitive data
   - Create `.streamlit/secrets.toml` (local) or use Streamlit Cloud secrets

2. **File Size Limits**
   - Streamlit Cloud: 1GB repo limit
   - Large model files should be downloaded on-demand or stored externally

3. **Rate Limiting**
   - Consider adding rate limits for public deployments
   - Use authentication if needed

---

## 🎯 Quick Start Commands

### Streamlit Cloud (Recommended)
```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Go to share.streamlit.io and deploy
```

### Local Testing Before Deployment
```bash
# Test your app locally first
streamlit run streamlit_app.py
```

---

## 📞 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Make sure all dependencies are in `requirements.txt`
   - Check that `src/` directory is included in repository

2. **Port Already in Use**
   - Change port: `streamlit run streamlit_app.py --server.port=8502`

3. **Memory Issues**
   - Reduce batch sizes in training
   - Use smaller models for demo

4. **Slow Loading**
   - Optimize imports
   - Use caching (`@st.cache_data`)
   - Consider using CDN for static assets

---

## 🎉 After Deployment

1. **Share your app URL** with users
2. **Monitor usage** via Streamlit Cloud dashboard
3. **Update regularly** by pushing to GitHub (auto-deploys)

---

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)

---

**Need Help?** Check the [Streamlit Community Forum](https://discuss.streamlit.io/) or open an issue on GitHub.

