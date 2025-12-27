# Deploy Pneumonia Detection API to Render

This guide walks you through deploying your FastAPI pneumonia detection backend to Render's free tier.

## Prerequisites

- [x] FastAPI application code
- [x] Trained model file (`pneumonia_detection_model.h5`)
- [ ] GitHub account
- [ ] Render account (sign up at https://render.com)

## Step 1: Prepare Your Repository

### 1.1 Initialize Git Repository (if not already done)

```bash
cd /home/alvee/Desktop/medical/api
git init
```

### 1.2 Add Model File to Repository

The model file is large (134MB), but Render supports it. Add it to your repo:

```bash
git add ../pneumonia_detection_model.h5
git add .
git commit -m "Initial commit: Pneumonia detection API"
```

### 1.3 Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `pneumonia-detection-api`)
3. **Do NOT** initialize with README (you already have code)
4. Copy the repository URL

### 1.4 Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection-api.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

### 2.1 Sign Up / Log In

1. Go to https://render.com
2. Sign up or log in (you can use your GitHub account)

### 2.2 Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account (if not already connected)
3. Select your repository: `pneumonia-detection-api`
4. Click **"Connect"**

### 2.3 Configure Service

Render will auto-detect the `render.yaml` file, but verify these settings:

- **Name**: `pneumonia-detection-api` (or your preferred name)
- **Runtime**: `Python`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Plan**: `Free`

### 2.4 Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. **Wait for deployment** (first deploy takes 5-10 minutes due to TensorFlow installation)

### 2.5 Monitor Deployment

- Watch the **Logs** tab to see installation progress
- Look for: `Model loaded successfully`
- Wait for status to change to **"Live"**

## Step 3: Get Your API Endpoint

Once deployed, Render provides a public URL:

```
https://pneumonia-detection-api-XXXX.onrender.com
```

- Copy this URL from the Render dashboard
- Your API will be accessible at this address

## Step 4: Test Your Deployment

### Test Health Check

```bash
curl https://your-app.onrender.com/
```

**Expected response:**
```json
{
  "status": "online",
  "model_loaded": true
}
```

### Test Model Stats

```bash
curl https://your-app.onrender.com/stats
```

**Expected response:**
```json
{
  "model_performance": {
    "accuracy": 0.8958,
    "precision": 0.8736,
    ...
  }
}
```

### Test Prediction (with an X-ray image)

```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -F "file=@/path/to/xray_image.jpg"
```

**Expected response:**
```json
{
  "prediction": "Pneumonia",
  "confidence": "92.34",
  "raw_probability": 0.9234,
  "visualization": "base64_encoded_image..."
}
```

## Step 5: Update Next.js Frontend

### 5.1 Update API URL

In your Next.js app (`/home/alvee/Desktop/medical/web-app`), update the API endpoint:

**Option A: Environment Variable (Recommended)**

Create or update `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://your-app.onrender.com
```

Then use it in your code:

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

**Option B: Direct Update**

Find where you make API calls and replace:

```typescript
// Before
const response = await fetch('http://localhost:8000/predict', ...);

// After
const response = await fetch('https://your-app.onrender.com/predict', ...);
```

### 5.2 Rebuild Frontend

```bash
cd /home/alvee/Desktop/medical/web-app
npm run build
npm run dev  # Or deploy to Vercel/Netlify
```

## Step 6: Test End-to-End

1. Open your Next.js app in browser
2. Upload a chest X-ray image
3. Verify prediction appears
4. Check Grad-CAM visualization (for pneumonia cases)

## Important Notes

### Free Tier Limitations

⚠️ **Service Sleep**: After 15 minutes of inactivity, the service sleeps
- First request after sleep takes 30-60 seconds (cold start)
- Subsequent requests are fast (~3-10 seconds)

⚠️ **CPU Only**: No GPU on free tier
- Inference is slower than local GPU
- Still acceptable for testing/demo purposes

⚠️ **Monthly Limit**: 750 hours/month
- Enough for development and light usage
- Resets every month

### Upgrade to Paid Tier ($7/month)

For production use, consider upgrading:

1. Go to your service in Render dashboard
2. Click **"Settings"** → **"Plan"**
3. Select **"Starter"** ($7/month)
4. Benefits:
   - No sleep/cold starts
   - Always-on service
   - Unlimited hours
   - Better performance

## Troubleshooting

### Build Fails

**Issue**: `ERROR: Could not install tensorflow`

**Solution**: TensorFlow requires specific Python version
- Ensure `PYTHON_VERSION=3.11.0` is set in environment variables
- Check Render build logs for specific error

### Model Not Loading

**Issue**: `Model is not loaded` error

**Solution**: Verify model file path
- Ensure `pneumonia_detection_model.h5` is in the repository
- Check the path in `main.py` (line 14): `MODEL_PATH = '../pneumonia_detection_model.h5'`
- Update if needed to: `MODEL_PATH = 'pneumonia_detection_model.h5'` (if in same directory)

### CORS Errors

**Issue**: Browser blocks requests from frontend

**Solution**: Already configured in `main.py` (lines 19-25)
- Verify `allow_origins=["*"]` is present
- For production, change to specific domain: `allow_origins=["https://yourdomain.com"]`

### Slow Response Times

**Issue**: Predictions take 30+ seconds

**Solution**:
- First request after sleep is slow (expected)
- Keep service warm with periodic health checks
- Or upgrade to paid tier

## Continuous Deployment

Render automatically deploys on every `git push`:

```bash
# Make changes to your code
git add .
git commit -m "Update model or API"
git push origin main

# Render will automatically rebuild and redeploy
```

## Monitoring

- **Logs**: Render Dashboard → Your Service → Logs
- **Metrics**: Dashboard shows CPU/Memory usage
- **Health Checks**: Render pings `/` endpoint every few minutes

## Next Steps

✅ API deployed to Render  
✅ Frontend updated with new endpoint  
✅ End-to-end testing complete  

**Ready for production?** Consider:
- [ ] Upgrade to paid tier for always-on service
- [ ] Add authentication (API keys)
- [ ] Set up custom domain
- [ ] Add request logging with Supabase
- [ ] Implement rate limiting
- [ ] Add monitoring/alerting

---

**Need Help?**
- Render Docs: https://render.com/docs
- Render Support: https://render.com/support
