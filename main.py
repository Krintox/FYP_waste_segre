import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('modelnew.h5')

# Load class labels (assuming you have saved them in a dictionary or file)
# Here, I'm assuming you've manually mapped class indices to labels
labels = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 3: 'Class 4', 4: 'Class 5', 5: 'Class 6'}

# Define a function to preprocess the image and make predictions
def prepare_image(img):
    img = img.resize((300, 300))  # Resize to match model input size
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return "Welcome to the Trash Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If the user did not select a file
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read the image file and preprocess it
    try:
        img = Image.open(io.BytesIO(file.read()))  # Open the image from the file object
        img_array = prepare_image(img)
        
        # Get model prediction
        prediction = model.predict(img_array)
        
        # Get the predicted class and probability
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = labels[predicted_class_idx]
        probability = np.max(prediction[0])
        
        return jsonify({
            'predicted_class': predicted_class,
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
