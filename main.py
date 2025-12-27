from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os

app = FastAPI()

# --- Configuration ---
MODEL_PATH = 'pneumonia_detection_model.h5'
IMG_SIZE = (224, 224)

# --- CORS Middleware ---
# Allow requests from the Next.js frontend (usually running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all. In production, specify domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model Variable ---
model = None

@app.on_event("startup")
def load_model():
    """Loads the model on startup."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        # Load model using the same methodology as predict.py
        # If using a custom optimized runtime (like ONNX) this would change, 
        # but for h5, keras load_model is standard.
        try:
             # GPU Memory Growth (Important if running API alongside training or other GPU tasks)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
            
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. Prediction endpoint will fail.")

# Model performance metrics (from evaluation)
MODEL_STATS = {
    "accuracy": 0.8958,
    "precision": 0.8736,
    "recall": 0.9744,
    "f1_score": 0.9212,
    "roc_auc": 0.9601,
    "specificity": 0.7650,
    "test_images": 624,
    "training_images": 5216
}

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.get("/stats")
def get_model_stats():
    """Return model performance statistics."""
    return {
        "model_performance": MODEL_STATS,
        "status": "Model trained on 5,216 chest X-rays",
        "version": "1.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict pneumonia from an uploaded image file.
    Returns prediction with Grad-CAM visualization.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. Read file into memory
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        original_image = pil_image.copy()  # Keep original for Grad-CAM
        
        # 2. Preprocessing (Must match training!)
        # Resize
        pil_image = pil_image.resize(IMG_SIZE)
        # Convert to array
        img_array = image.img_to_array(pil_image)
        # Expand dims
        img_array = np.expand_dims(img_array, axis=0)
        # Normalize (Rescale 1/255)
        img_array /= 255.0

        # 3. Inference
        prediction_prob = model.predict(img_array)
        confidence = float(prediction_prob[0][0])

        # 4. Interpretation
        if confidence > 0.5:
            label = "Pneumonia"
            # Confidence is probability of class 1 (Pneumonia)
            conf_percent = confidence * 100
        else:
            label = "Normal"
            # Confidence for Normal is 1 - prob(Pneumonia)
            conf_percent = (1 - confidence) * 100

        # 5. Generate visualization (only for Pneumonia predictions)
        visualization_base64 = None
        if confidence > 0.5:  # Only generate for pneumonia cases
            from simple_viz import generate_visualization
            import cv2
            import base64
            
            try:
                # Generate heatmap overlay
                visualization = generate_visualization(model, img_array, pil_image)
                
                # Convert to base64 for transmission
                _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                visualization_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as viz_error:
                print(f"Visualization generation failed: {viz_error}")
                # Continue without visualization rather than failing
                visualization_base64 = None

        return {
            "prediction": label,
            "confidence": f"{conf_percent:.2f}",
            "raw_probability": confidence,
            "visualization": visualization_base64  # Base64 encoded heatmap image
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
