import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model

def simple_activation_heatmap(model, img_array):
    """
    Generate heatmap from activation maps weighted by prediction.
    Shows regions that contribute to pneumonia detection.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image (1, 224, 224, 3)
        
    Returns:
        heatmap: Numpy array (H, W) normalized 0-1
    """
    # Get the prediction first
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Find the last convolutional layer
    conv_layer = None
    conv_layer_index = None
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name.lower():
            conv_layer = layer
            conv_layer_index = i
    
    if conv_layer is None:
        return np.zeros((224, 224))
    
    # Create a model that outputs the conv layer activations
    from tensorflow.keras.models import Sequential as SequentialModel
    activation_model = SequentialModel()
    for i, layer in enumerate(model.layers):
        activation_model.add(layer)
        if i == conv_layer_index:
            break
    
    # Get activations
    activations = activation_model.predict(img_array, verbose=0)
    
    # Weight channels by their variance (more active channels get more weight)
    # This helps highlight regions that matter for the detection
    channel_weights = np.var(activations[0], axis=(0, 1))
    
    # If predicting pneumonia (>0.5), use positive weighting
    # Otherwise invert for normal cases
    if prediction > 0.5:
        # For pneumonia: emphasize high-activation regions
        heatmap = np.sum(activations[0] * channel_weights, axis=-1)
    else:
        # For normal: show inverted (low activation = healthy)
        heatmap = np.sum(activations[0] * channel_weights, axis=-1)
        heatmap = np.max(heatmap) - heatmap
    
    # Normalize to 0-1
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    
    # Apply power transform to enhance contrast
    heatmap = np.power(heatmap, 1.5)
    
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image (PIL or numpy)
        heatmap: Heatmap array (H, W)
        alpha: Transparency
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image (numpy array RGB)
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'convert'):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    # Ensure uint8
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Convert heatmap to colored version
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    output = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    return output


def generate_visualization(model, img_array, original_image):
    """
    Complete pipeline to generate visualization.
    
    Args:
        model: Trained model
        img_array: Preprocessed image (1, 224, 224, 3)
        original_image: Original PIL image
        
    Returns:
        Visualization image (numpy RGB)
    """
    heatmap = simple_activation_heatmap(model, img_array)
    visualization = overlay_heatmap(original_image, heatmap, alpha=0.6)
    return visualization
