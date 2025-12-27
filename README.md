<div align="center">

# 🫁 Pneumonia Detection API

**AI-Powered Chest X-Ray Analysis with Grad-CAM Visualization**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127.1-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com)

*State-of-the-art deep learning model for pneumonia detection with explainable AI visualization*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [API Docs](#-api-documentation) • [Deploy](#-deployment)

</div>

---

## 🎯 Overview

A production-ready FastAPI backend that uses deep learning to detect pneumonia from chest X-ray images. The model achieves **89.58% accuracy** and includes Grad-CAM visualization to highlight affected lung regions, making AI predictions explainable and trustworthy for medical professionals.

### Key Highlights

- 🧠 **Deep Learning Model**: Custom CNN trained on 5,216 chest X-ray images
- 🎨 **Grad-CAM Visualization**: Heatmap overlay showing pneumonia-affected areas
- ⚡ **High Performance**: 89.58% accuracy, 97.44% recall, 96.01% ROC-AUC
- 🚀 **Production Ready**: FastAPI with CORS, health checks, and comprehensive error handling
- 📊 **Model Metrics**: Built-in endpoint for performance statistics
- 🔒 **Type Safe**: Full type hints and validation with Pydantic
- 📦 **Easy Deployment**: One-click deploy to Render with pre-configured YAML

---

## ✨ Features

### 🔬 Medical AI Features
- **Pneumonia Detection**: Binary classification (Normal vs Pneumonia)
- **Confidence Scores**: Probability-based predictions with percentage confidence
- **Explainable AI**: Grad-CAM heatmap visualization for pneumonia cases
- **Medical-Grade Accuracy**: Trained on professionally curated chest X-ray dataset

### 🛠️ Technical Features
- **RESTful API**: Clean, well-documented FastAPI endpoints
- **Image Processing**: Supports JPEG, PNG, and other common image formats
- **CORS Enabled**: Ready for web frontend integration
- **Health Monitoring**: Status endpoint for uptime checks
- **Performance Metrics**: Accessible model statistics endpoint
- **Auto-scaling**: Ready for cloud deployment with Render

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 89.58% |
| **Precision** | 87.36% |
| **Recall** | 97.44% |
| **F1-Score** | 92.12% |
| **ROC-AUC** | 96.01% |
| **Specificity** | 76.50% |

**Training Data**: 5,216 chest X-ray images  
**Test Data**: 624 images  
**Model Architecture**: Custom Convolutional Neural Network  
**Framework**: TensorFlow 2.20 / Keras 3.13

---

## 🎬 Demo

### API Response Example

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@chest_xray.jpg"
```

**Response**:
```json
{
  "prediction": "Pneumonia",
  "confidence": "92.34",
  "raw_probability": 0.9234,
  "visualization": "base64_encoded_heatmap_image..."
}
```

### Grad-CAM Visualization

The API generates heatmap overlays showing which lung regions influenced the pneumonia prediction, helping medical professionals understand the AI's decision-making process.

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip package manager
- 512MB+ RAM (2GB+ recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pneumonia-detection-api.git
   cd pneumonia-detection-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **⚠️ Obtain the model file**
   
   The trained model file (128MB) is **not included in this repository** due to GitHub's file size limit.
   
   See [MODEL_STORAGE.md](MODEL_STORAGE.md) for instructions on how to obtain the model file.
   
   Place `pneumonia_detection_model.h5` in the same directory as `main.py`.

5. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

---

## 📚 API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response**:
```json
{
  "status": "online",
  "model_loaded": true
}
```

---

#### `GET /stats`
Get model performance metrics.

**Response**:
```json
{
  "model_performance": {
    "accuracy": 0.8958,
    "precision": 0.8736,
    "recall": 0.9744,
    "f1_score": 0.9212,
    "roc_auc": 0.9601,
    "specificity": 0.7650,
    "test_images": 624,
    "training_images": 5216
  },
  "status": "Model trained on 5,216 chest X-rays",
  "version": "1.0"
}
```

---

#### `POST /predict`
Predict pneumonia from chest X-ray image.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (image file)

**Response**:
```json
{
  "prediction": "Pneumonia" | "Normal",
  "confidence": "92.34",
  "raw_probability": 0.9234,
  "visualization": "base64_encoded_image" | null
}
```

**Notes**:
- Visualization is only generated for pneumonia predictions
- Accepted formats: JPEG, PNG, BMP, TIFF
- Image is automatically resized to 224x224

---

### Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: `/docs` - Try out API endpoints interactively
- **ReDoc**: `/redoc` - Beautiful API reference documentation

---

## 🌐 Deployment

### Deploy to Render (Free Tier)

This repository is pre-configured for Render deployment with `render.yaml`.

1. **Push to GitHub** (see instructions below)

2. **Sign up/Login to Render**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Select your repository
   - Render auto-detects configuration

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - Get your public URL: `https://your-app.onrender.com`

**Free Tier Limitations**:
- Service sleeps after 15 min inactivity (30-60s cold start)
- 750 hours/month
- CPU only (no GPU)

**Upgrade to Paid** ($7/month):
- Always-on (no cold starts)
- Better performance
- Unlimited hours

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 🏗️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI 0.127.1 |
| **Server** | Uvicorn 0.40.0 |
| **ML Framework** | TensorFlow 2.20, Keras 3.13 |
| **Image Processing** | Pillow 12.0, OpenCV 4.12 |
| **Numerical Computing** | NumPy 2.2.6 |
| **Validation** | Pydantic 2.12.5 |
| **Runtime** | Python 3.11 |

---

## 📁 Project Structure

```
pneumonia-detection-api/
├── main.py                          # FastAPI application & endpoints
├── simple_viz.py                    # Grad-CAM visualization generator
├── gradcam.py                       # Grad-CAM implementation
├── pneumonia_detection_model.h5     # Trained TensorFlow model (128MB)
├── requirements.txt                 # Python dependencies
├── render.yaml                      # Render deployment config
├── DEPLOYMENT.md                    # Detailed deployment guide
└── README.md                        # This file
```

---

## 🔧 Configuration

### Environment Variables

None required for basic operation. Optional configurations:

- `PORT`: Server port (default: 8000, auto-set by Render)
- `PYTHON_VERSION`: Python version (default: 3.11.0)

### Model Configuration

In `main.py`:
```python
MODEL_PATH = 'pneumonia_detection_model.h5'  # Model file path
IMG_SIZE = (224, 224)                         # Input image size
```

### CORS Configuration

Currently allows all origins for development. For production, update in `main.py`:
```python
allow_origins=["https://yourdomain.com"]  # Restrict to your frontend domain
```

---

## 🧪 Testing

### Manual Testing with cURL

**Health Check**:
```bash
curl http://localhost:8000/
```

**Get Statistics**:
```bash
curl http://localhost:8000/stats
```

**Predict Pneumonia**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/chest_xray.jpg"
```

### Testing with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Prediction
with open("chest_xray.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

---

## 🎓 How It Works

### 1. Image Preprocessing
- Resize to 224×224 pixels
- Convert to RGB array
- Normalize pixel values (0-1 range)

### 2. Model Inference
- Feed preprocessed image to CNN
- Output: Probability score (0-1)
- >0.5 = Pneumonia, <0.5 = Normal

### 3. Grad-CAM Visualization
- Analyze last convolutional layer activations
- Generate heatmap of important regions
- Overlay heatmap on original X-ray
- Encode as base64 for API response

### 4. Response Formatting
- Classification label (Pneumonia/Normal)
- Confidence percentage
- Raw probability score
- Visualization (if pneumonia detected)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add authentication/API keys
- [ ] Implement rate limiting
- [ ] Add request logging with database
- [ ] Support batch predictions
- [ ] Add more visualization options
- [ ] Implement A/B testing for model versions
- [ ] Add prometheus metrics
- [ ] Create Docker container
- [ ] Add unit tests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This is a research/demonstration project and should **not** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

## 📞 Support

- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **API Docs**: `/docs` endpoint when running
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/pneumonia-detection-api/issues)

---

## 🌟 Acknowledgments

- Dataset curated from publicly available chest X-ray datasets
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [TensorFlow](https://www.tensorflow.org/)
- Grad-CAM implementation inspired by the original [paper](https://arxiv.org/abs/1610.02391)

---

<div align="center">

**Made with ❤️ for better healthcare through AI**

⭐ Star this repo if you find it helpful!

</div>
