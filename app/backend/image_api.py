import sys
from flask import Flask, request, jsonify
import os

# Fix Python path so `src/` is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from src.image_forensics.image_predictor import ImagePredictor

app = Flask(__name__)
predictor = ImagePredictor()

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    temp_path = os.path.join("temp_uploads", image_file.filename)
    os.makedirs("temp_uploads", exist_ok=True)
    image_file.save(temp_path)
    
    try:
        result = predictor.predict(temp_path)
        # Ensure we return relative paths for URLs if needed, but here we just return absolute for now
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
