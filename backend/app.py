import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering plots

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import csv
import logging

# Custom imports for stroke-to-image conversion
from strokes_to_image import csv_to_strokes, strokes_to_image, draw_circle_based_on_input, draw_triangle_based_on_input

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Load the pre-trained model
model = load_model('/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/saved_model.keras')

@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files such as CSS, JS, and images."""
    return send_from_directory(app.static_folder, path)

def encode_image(img_array):
    """Convert a NumPy array to a base64 encoded PNG."""
    img = Image.fromarray(img_array.astype('uint8'), 'L')  # Ensure proper format
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_csv(strokes):
    """Generate a CSV string from strokes."""
    output = io.StringIO()
    writer = csv.writer(output)
    for stroke in strokes:
        writer.writerow(stroke)
    return output.getvalue()

def preprocess_image(image):
    """Convert grayscale image to RGB, resize, normalize, and add batch dimension."""
    # Convert grayscale to RGB
    image_rgb = np.stack([image] * 3, axis=-1)
    
    # Resize the image to 32x32
    image_rgb = Image.fromarray(image_rgb)
    image_rgb = image_rgb.resize((32, 32))
    
    # Convert the image back to a NumPy array
    image_rgb = np.array(image_rgb)
    
    # Normalize the image
    image_rgb = image_rgb / 255.0
    
    # Add batch dimension
    image_rgb = np.expand_dims(image_rgb, axis=0)
    
    return image_rgb

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if data is provided
        if 'csv_data' in data:
            csv_data = data['csv_data']
            strokes = csv_to_strokes(csv_data)
        elif 'strokes' in data:
            strokes = data['strokes']
        else:
            return jsonify({'error': 'No data provided'}), 400

        # Convert strokes to a more regularized image
        img_array = strokes_to_image(strokes)

        # Process image for model prediction
        img_array_model = preprocess_image(img_array)

        prediction = model.predict(img_array_model)
        predicted_digit = np.argmax(prediction)

        # Generate a dynamic shape based on prediction
        if predicted_digit == 0:  # Assuming 0 corresponds to Circle
            regular_image = draw_circle_based_on_input(strokes)
        elif predicted_digit == 1:  # Assuming 1 corresponds to Triangle
            regular_image = draw_triangle_based_on_input(strokes)
        else:
            # Handle other cases or fallback
            regular_image = img_array

        # Create and save a plot of the regularized image
        plt.figure(figsize=(3, 3))
        plt.imshow(regular_image, cmap='gray')
        plt.axis('off')

        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
        img_bytes.seek(0)

        # Encode image to base64
        img_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        # Generate CSV output
        csv_output = generate_csv(strokes)

        return jsonify({'prediction': int(predicted_digit), 'image': img_str, 'csv': csv_output})
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
    app.run(host='0.0.0.0', port=5001)
