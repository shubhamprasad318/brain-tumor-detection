from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import time
from PIL import Image

app = Flask(__name__)

# Model paths
MODEL_PATH_1 = "models/best_cnn.keras"
MODEL_PATH_2 = "models/best_mobilenet.keras"

# Load models
model1 = tf.keras.models.load_model(MODEL_PATH_1)
model2 = tf.keras.models.load_model(MODEL_PATH_2)

# Automatically detect input sizes
target_size1 = model1.input_shape[1:3]
target_size2 = model2.input_shape[1:3]

# Upload & heatmap folders
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, HEATMAP_FOLDER=HEATMAP_FOLDER)


def preprocess_image(path, size):
    img = image.load_img(path, target_size=size)
    array = image.img_to_array(img) / 255.0
    return np.expand_dims(array, axis=0)


def is_mri_image(path, threshold=15, ratio_tol=0.5):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if abs(w - h) / max(w, h) > ratio_tol:
        pass
    arr = np.array(img)
    diffs = [np.mean(np.abs(arr[:, :, i] - arr[:, :, j]))
             for i in range(3) for j in range(i+1, 3)]
    mean_diff = float(np.mean(diffs))
    return (mean_diff < threshold), mean_diff


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv_pw_13_relu"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_encode_heatmap(original_path, heatmap, alpha=0.4):
    img = Image.open(original_path)
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize(img.size)
    heatmap_img = heatmap_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size)
    overlay.paste(heatmap_img, (0, 0), heatmap_img)
    combined = Image.alpha_composite(img.convert("RGBA"), overlay)

    filename = f"heat_{int(time.time())}.png"
    path = os.path.join(app.config['HEATMAP_FOLDER'], filename)
    combined.save(path)
    return path


def ensemble_predict(path, w1=0.5, w2=0.5, gen_heatmap=False):
    arr1 = preprocess_image(path, target_size1)
    arr2 = preprocess_image(path, target_size2)
    p1 = float(model1.predict(arr1)[0][0])
    p2 = float(model2.predict(arr2)[0][0])
    final = w1 * p1 + w2 * p2
    label = "Brain Tumor Detected" if final > 0.5 else "No Brain Tumor Detected"
    report = {
        "model1_confidence": round(p1 * 100, 2),
        "model2_confidence": round(p2 * 100, 2),
        "ensemble_confidence": round(final * 100, 2),
        "prediction": label
    }
    heatmap_url = None
    if gen_heatmap:
        try:
            heatmap = make_gradcam_heatmap(arr2, model2)
            heatmap_path = save_and_encode_heatmap(path, heatmap)
            heatmap_url = heatmap_path.replace(app.root_path + os.sep, '')
        except Exception:
            heatmap_url = None
    return report, heatmap_url


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = f"{int(time.time())}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    is_mri, diff = is_mri_image(save_path)
    if not is_mri:
        return jsonify({'error': 'Not an MRI scan', 'diff': diff}), 400

    # optional weights via form or JSON
    w1 = float(request.form.get('w1', 0.5))
    w2 = float(request.form.get('w2', 0.5))
    gen_heatmap = request.form.get('heatmap', 'false').lower() == 'true'

    try:
        report, heat_url = ensemble_predict(save_path, w1, w2, gen_heatmap)
        response = {
            'report': report,
            'image_url': f"/static/uploads/{filename}"
        }
        if heat_url:
            response['heatmap_url'] = f"/{heat_url}"
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Static file routes
@app.route('/static/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/heatmaps/<path:filename>')
def heatmaps(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
