# 🚀 Ollama Setup Guide for Q&A Bot

## 🔍 **Current Problem:**
Your app is deployed but Ollama model is not working because the `MODEL_URL` needs a real Ollama endpoint.

## 🌐 **Solution Options:**

### **Option 1: Self-Hosted Ollama (Recommended - Most Reliable)**

#### **Step 1: Get a VPS/Server**
- **DigitalOcean**: $5/month droplet
- **Linode**: $5/month server  
- **Vultr**: $3.50/month instance
- **AWS EC2**: Free tier available

#### **Step 2: Install Ollama on Your Server**
```bash
# Connect to your server
ssh root@your-server-ip

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in new terminal)
ollama pull llama2
# or
ollama pull gemma:2b
```

#### **Step 3: Make it Accessible via Nginx**
```bash
# Install nginx
apt update && apt install nginx

# Create nginx config
nano /etc/nginx/sites-available/ollama
```

**Add this configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;  # or your-server-ip
    
    location /api/ {
        proxy_pass http://localhost:11434;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

```bash
# Enable site
ln -s /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Test
curl http://your-domain.com/api/tags
```

#### **Step 4: Update Your render.yaml**
```yaml
- key: MODEL_URL
  value: https://your-domain.com/api/generate
```

### **Option 2: Use Ollama Cloud (Paid but Easy)**

1. **Go to [Ollama Cloud](https://ollama.ai/cloud)**
2. **Create account**
3. **Get your API endpoint**
4. **Update MODEL_URL in Render Dashboard**

### **Option 3: Use RunPod (GPU-Powered)**

1. **Go to [RunPod](https://runpod.io/)**
2. **Create account**
3. **Deploy Ollama template**
4. **Get your endpoint**
5. **Update MODEL_URL**

### **Option 4: Test with Free API (Temporary)**

**Update your render.yaml:**
```yaml
- key: MODEL_URL
  value: https://api.ollama.ai/v1/generate
- key: MODEL_NAME
  value: llama2
```

## 🔧 **Quick Fix for Now:**

### **Update in Render Dashboard:**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click on your `qa-bot` service
3. Go to **Environment** tab
4. Update these variables:

```
MODEL_URL = https://api.ollama.ai/v1/generate
MODEL_NAME = llama2
```

### **Test the Connection:**
Visit: `https://qa-bot-kd79.onrender.com/health`

This should show:
```json
{
  "ollama_status": "healthy",
  "model_name": "llama2"
}
```

## 🚨 **Important Notes:**

1. **Official Ollama API** (`api.ollama.ai`) requires:
   - Valid API key (if you have one)
   - Rate limiting may apply
   - Not free for production use

2. **Self-hosted Ollama** is:
   - Most reliable
   - Completely free
   - Full control over models
   - No rate limits

3. **Free APIs** are:
   - Unreliable for production
   - May have rate limits
   - Can be slow

## 🎯 **Recommended Action:**

1. **Immediate**: Update to `https://api.ollama.ai/v1/generate` for testing
2. **Short-term**: Set up self-hosted Ollama on a VPS
3. **Long-term**: Use Ollama Cloud for production

## 📞 **Need Help?**

- **Self-hosted setup**: I can guide you step-by-step
- **Cloud options**: I can help you choose the best one
- **Testing**: I can help you verify the connection

---

**Choose your preferred option and I'll help you implement it! 🚀**
