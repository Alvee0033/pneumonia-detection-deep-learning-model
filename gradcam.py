import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_2'):
    """
    Generate Grad-CAM heatmap for the given image.
    
    Args:
        img_array: Preprocessed image array (1, 224, 224, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        
    Returns:
        heatmap: Numpy array (224, 224) with values 0-255
    """
    # First, ensure model is built by calling it once
    _ = model(img_array)
    
    # Get the last conv layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except:
        # If layer not found, try to find any conv layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                last_conv_layer_name = layer.name
                break
    
    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        # For binary classification, get the probability
        class_channel = preds[0][0]  # Sigmoid output
    
    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the predicted class
    layer_output = layer_output[0]
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    
    return heatmap

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    
    Args:
        img: Original PIL Image or numpy array
        heatmap: Heatmap array (H, W)
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
        
    Returns:
        superimposed_img: Numpy array (H, W, 3) in RGB
    """
    # Convert PIL Image to numpy if needed
    if hasattr(img, 'convert'):
        img = np.array(img.convert('RGB'))
    
    # Ensure img is uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed_img

def generate_gradcam_visualization(model, img_array, original_img):
    """
    Complete pipeline to generate Grad-CAM visualization.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        original_img: Original PIL Image
        
    Returns:
        visualization: Numpy array (224, 224, 3) with heatmap overlay
    """
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    
    # Overlay on original image
    visualization = overlay_heatmap_on_image(original_img, heatmap, alpha=0.5)
    
    return visualization
