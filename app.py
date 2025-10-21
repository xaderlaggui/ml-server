from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Model and directories
MODEL_PATH = 'braille_model.h5'
TRAIN_DIR = './Braille Dataset/train'
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# Data preprocessing for training (for class label mapping)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate class-to-label mapping
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=42,
    class_mode='categorical'
)
class_labels = train_generator.class_indices
labels_map = {v: k for k, v in class_labels.items()}  # Reverse mapping


# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize pixels
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Prediction endpoint
@app.route('/predict_braille', methods=['POST'])
def predict_braille():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Save the uploaded image temporarily
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    try:
        # Preprocess the image and predict
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class_index = tf.argmax(predictions, axis=-1).numpy()[0]
        predicted_label = labels_map.get(predicted_class_index, "Unknown")

        # Return prediction
        return jsonify({
            'predicted_class': predicted_label,
            'confidence': float(predictions[0][predicted_class_index])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(image_path)  # Clean up uploaded file


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
