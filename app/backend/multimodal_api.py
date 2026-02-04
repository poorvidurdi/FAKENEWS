import os
import sys
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# -------------------------------------------------
# Fix Python path so `src/` is importable
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

# -------------------------------------------------
from src.multimodal.multimodal_predictor import predict_multimodal
from src.image_forensics.image_predictor import ImagePredictor

# Initialize ImagePredictor once
image_forensics = ImagePredictor()

# -------------------------------------------------
# Flask app setup
# -------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# Health check (optional but useful)
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "multimodal backend running"}), 200

# -------------------------------------------------
# Multimodal prediction endpoint
# -------------------------------------------------
@app.route("/predict_multimodal", methods=["POST"])
def predict_multimodal_route():
    try:
        # ----------------------------
        # Validate inputs
        # ----------------------------
        if "text" not in request.form:
            return jsonify({"error": "Missing text field"}), 400

        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        text = request.form["text"]
        image = request.files["image"]

        if not text.strip():
            return jsonify({"error": "Text is empty"}), 400

        # ----------------------------
        # Save image temporarily
        # ----------------------------
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)

        # ----------------------------
        # Run forensics & multimodal prediction
        # ----------------------------
        forensics_result = image_forensics.predict(image_path)
        result = predict_multimodal(text, image_path, forensics_result=forensics_result)
        
        # Ensure forensics detail is included in result for UI
        result["image_forensics"] = forensics_result

        return jsonify(result), 200

    except Exception as e:
        # 🔴 PRINT FULL TRACEBACK (VERY IMPORTANT)
        traceback.print_exc()

        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

    finally:
        # ----------------------------
        # Cleanup uploaded image
        # ----------------------------
        try:
            if "image_path" in locals() and os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

# -------------------------------------------------
# Run backend
# -------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5002,
        debug=True
    )

