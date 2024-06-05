from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
import os
import pickle
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = 'face_recognition_model.pkl'

# Load the trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load the model when the application starts
representations = load_model(MODEL_PATH)

@app.route('/')
def index():
    return send_from_directory('', 'index.html')  # Serve the HTML file


@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Check if the 'file' field is present in the form data
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 401

        # Get the image file from the request
        image_file = request.files['file']

        # Read the image file and convert it to a numpy array
        image = Image.open(image_file)
        image_np = np.array(image)

        # Perform face recognition using the loaded model
        input_representation = DeepFace.represent(image_np, model_name='Facenet')[0]['embedding']
        
        min_distance = float('inf')
        name = None
        for rep, rep_name in representations:
            similarity = cosine_similarity([input_representation], [rep])[0][0]
            if similarity < min_distance:
                min_distance = similarity
                name = rep_name

        if name is None:
            return jsonify({"error": "No matching faces found"}), 403
        else:
            return jsonify({"name": name}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/save_photos', methods=['POST'])
def save_photos():
    try:
        # Check if the 'name' field is present in the form data
        name = request.form.get('name')
        if not name:
            return jsonify({"error": "Name is required"}), 400

        # Create a directory for the person if it doesn't exist
        person_dir = os.path.join('known_faces', name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Get the image files from the request
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files part"}), 401

        for file in files:
            # Save each file to the person's directory
            file_path = os.path.join(person_dir, file.filename)
            file.save(file_path)

        # Retrain and save the model after saving new photos
        os.system('python trainer.py')
        global representations
        representations = load_model(MODEL_PATH)

        return jsonify({"message": "Photos saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
