from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load known faces
def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                known_faces.append(image_path)
                known_names.append(name)
    return known_faces, known_names

known_faces, known_names = load_known_faces('known_faces')

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

        # Perform face recognition using deepface
        results = DeepFace.find(img_path=image_file, db_path='known_faces', enforce_detection=False)

        if results.empty:
            return jsonify({"error": "No matching faces found"}), 403
        else:
            name = results.iloc[0]['identity'].split('/')[-1]
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

        # Reload known faces after saving new photos
        global known_faces, known_names
        known_faces, known_names = load_known_faces('known_faces')

        return jsonify({"message": "Photos saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
