from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import time

app = Flask(__name__)

# Model paths
MODEL_PATH_1 = "D:/brain_tumor_project/models/best_model.keras"
MODEL_PATH_2 = "D:/brain_tumor_project/models/best_modelwithtransferlearning.keras"

# Load models
model1 = tf.keras.models.load_model(MODEL_PATH_1)
model2 = tf.keras.models.load_model(MODEL_PATH_2)

# Automatically detect input sizes
input_size1 = model1.input_shape[1:3]  # (height, width) of model1
input_size2 = model2.input_shape[1:3]  # (height, width) of model2

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(img_path, target_size):
    """Preprocess image based on target size."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def ensemble_predict(file_path):
    """Predict using both models and combine results."""
    img_array1 = preprocess_image(file_path, target_size=input_size1)  # Auto-detected size
    img_array2 = preprocess_image(file_path, target_size=input_size2)  # Auto-detected size

    pred1 = model1.predict(img_array1)[0][0]  # Prediction from Model 1
    pred2 = model2.predict(img_array2)[0][0]  # Prediction from Model 2

    # Simple average ensemble
    final_prediction = (pred1 + pred2) / 2  
    confidence = round(final_prediction * 100, 2)

    # Final Decision
    result = "Brain Tumor Detected" if final_prediction > 0.5 else "No Brain Tumor Detected"

    return result, confidence

@app.route("/", methods=["GET"])
def home():
    """Serve the frontend HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate unique filename
    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        result, confidence = ensemble_predict(file_path)
        return jsonify({"result": result, "confidence": confidence, "image_url": file_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
