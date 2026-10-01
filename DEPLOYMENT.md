# 🚀 Live Deployment Guide for CERTUS-S2

This guide walks you through deploying **CERTUS-S2** to get a permanent, live public web link (e.g. `https://certus-s2.streamlit.app`).

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended — Free & 1-Click)

Streamlit Community Cloud hosts Streamlit apps for free directly from your GitHub repository with automatic updates on every `git push`.

### Step 1: Push your latest changes to GitHub
In your local terminal:
```bash
git add .
git commit -m "Configure production deployment for Streamlit Cloud"
git push origin main
```

### Step 2: Open Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** (or [streamlit.io/cloud](https://streamlit.io/cloud)).
2. Sign in with your GitHub account: `Supercalifragilisticexpialidociouscoder`.

### Step 3: Deploy New App
1. Click the **"New app"** button in your dashboard.
2. Fill in the app settings:
   - **Repository:** `Supercalifragilisticexpialidociouscoder/SIH26142`
   - **Branch:** `main`
   - **Main file path:** `app.py` (or `certus-s2/ui/app.py`)
   - **App URL (subdomain):** Choose your custom URL (e.g. `certus-s2` $\rightarrow$ gives `https://certus-s2.streamlit.app`)

### Step 4: Configure Secrets (CDSE Credentials)
1. In the same deployment modal, click **"Advanced settings..."**
2. In the **Secrets** text box, paste your CDSE credentials:
```toml
CDSE_CLIENT_ID = "sh-d93a9bc6-87d8-4a88-b7f5-36d714b1d139"
CDSE_CLIENT_SECRET = "spP2i3QUgvEvKTTkm3NRA6Vlb8coR9fe"
```
3. Python version: Select **3.10** or **3.11**.

### Step 5: Click "Deploy!"
Streamlit Cloud will provision the container, install packages from `requirements.txt` and `packages.txt`, and launch your app.
Your app will be live at:
$$\text{https://<your-subdomain>.streamlit.app}$$

---

## ⚡ Method 2: Hugging Face Spaces (Alternative — 16 GB Free RAM)

If you need dedicated GPU or 16GB free RAM for large AI model inference:

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Space name: `certus-s2`
3. Space SDK: Select **Streamlit** (or **Docker**).
4. Connect or push this repository to your Hugging Face Space.
5. In **Space Settings $\rightarrow$ Variables and secrets**, add:
   - `CDSE_CLIENT_ID`
   - `CDSE_CLIENT_SECRET`
6. Your live link will be: `https://huggingface.co/spaces/<your-username>/certus-s2`.

---

## 🛠️ Deployment Configuration Summary

| File | Purpose |
| :--- | :--- |
| [`app.py`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/app.py) | Root entry point forwarding directly to CERTUS-S2 UI |
| [`requirements.txt`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/requirements.txt) | Python dependencies (PyTorch, Rasterio, Streamlit, etc.) |
| [`packages.txt`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/packages.txt) | Debian system libraries (`gdal-bin`, `libgdal-dev`) |
| [`.streamlit/config.toml`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/.streamlit/config.toml) | Dark theme tokens, production server settings, port configuration |
| [`Dockerfile`](file:///Users/sripranavireddypalle/Documents/GitHub/SIH26142/Dockerfile) | Production Docker container configuration |
