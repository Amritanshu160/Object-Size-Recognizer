# app.py

import os

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained object detection model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Create a directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_images(image_paths):
    return [cv2.imread(str(path)) for path in image_paths]

def detect_objects_with_tensorflow(image):
    image_resized = cv2.resize(image, (320, 320))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)

    # Perform the detection
    results = model(image_tensor)
    detections = {key: value.numpy() for key, value in results.items()}

    return detections, image_rgb

def extract_dimensions_from_results(detections, image):
    height, width, _ = image.shape

    for i in range(int(detections['num_detections'][0])):
        score = detections['detection_scores'][0][i]
        if score < 0.3:
            continue

        box = detections['detection_boxes'][0][i]
        y1, x1, y2, x2 = box
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Length: {x2 - x1}, Width: {y2 - y1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        length = x2 - x1
        width = y2 - y1
        height = 10  # Assumed height; adjust if 3D reconstruction is needed
        return length, width, height, image

    return 0, 0, 0, image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files[]")
    image_paths = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        image_paths.append(Path(file_path))

    images = load_images(image_paths)
    total_length, total_width, total_height = 0, 0, 0
    results_images = []

    for image in images:
        detections, processed_image = detect_objects_with_tensorflow(image)
        length, width, height, annotated_image = extract_dimensions_from_results(detections, processed_image)

        total_length += length
        total_width += width
        total_height += height
        results_images.append(annotated_image)

    avg_length = total_length / len(images) if images else 0
    avg_width = total_width / len(images) if images else 0
    avg_height = total_height / len(images) if images else 0
    volume = avg_length * avg_width * avg_height

    results = {
        "length": avg_length,
        "width": avg_width,
        "height": avg_height,
        "volume": volume,
    }

    output_files = []
    for idx, result_image in enumerate(results_images):
        output_path = f'static/result_{idx}.jpg'
        cv2.imwrite(output_path, result_image)
        output_files.append(output_path)

    results["images"] = output_files

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
