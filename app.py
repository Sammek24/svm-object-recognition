"""
SVM Image-Based Object Recognition Web Application
===================================================
Flask web server running Python-based SVM training, feature extraction (HOG),
real-time inference, and interactive visualization studio.
"""

import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS

from svm_engine import engine

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Train a default model on startup (Universal: Numbers + Alphabets + Objects)
print("[INFO] Initializing SVM Universal Vision Engine (Numbers + Alphabets + Objects)...")
engine.train(dataset_name="universal", feature_type="hog", kernel="rbf", C=1.0, gamma="scale")
print("[INFO] Universal SVM Model initialized successfully.")


@app.route("/")
def index():
    """Render the main web application UI."""
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Train an SVM model with specified parameters.
    Request JSON:
      - dataset: 'digits' | 'shapes' | 'vehicles_vs_animals'
      - feature_type: 'hog' | 'color_hist' | 'raw' | 'hog_color'
      - kernel: 'rbf' | 'linear' | 'poly'
      - C: float
      - gamma: 'scale' | float
      - degree: int
      - grid_search: bool
    """
    try:
        data = request.get_json() or {}
        dataset = data.get("dataset", "digits")
        feature_type = data.get("feature_type", "hog")
        kernel = data.get("kernel", "rbf")
        c_param = float(data.get("C", 1.0))
        gamma_param = data.get("gamma", "scale")
        degree_param = int(data.get("degree", 3))
        use_grid_search = bool(data.get("grid_search", False))

        metrics = engine.train(
            dataset_name=dataset,
            feature_type=feature_type,
            kernel=kernel,
            C=c_param,
            gamma=gamma_param,
            degree=degree_param,
            use_grid_search=use_grid_search
        )
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict object class from uploaded image or canvas base64 image.
    """
    try:
        data = request.get_json() or {}
        image_data = data.get("image")

        if not image_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Strip base64 header if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy OpenCV format
        img_np = np.array(pil_img)

        # Handle RGBA / RGB / Grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:
            # Drawing canvas is often transparent black/white. Composite onto black background
            bg = Image.new("RGB", pil_img.size, (0, 0, 0))
            bg.paste(pil_img, mask=pil_img.split()[3])
            img_np = np.array(bg)

        # Run inference through SVM engine
        source = data.get("source", "canvas")
        result = engine.predict_image(img_np, source=source)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/generate-python-script", methods=["POST"])
def api_generate_script():
    """
    Generate clean, production-ready Scikit-Learn Python script
    based on current web configuration.
    """
    data = request.get_json() or {}
    dataset = data.get("dataset", "digits")
    feature = data.get("feature_type", "hog")
    kernel = data.get("kernel", "rbf")
    c_val = data.get("C", 1.0)
    grid_search = data.get("grid_search", False)

    code_template = f'''"""
Image-Based Object Recognition using Support Vector Machines (SVM)
==================================================================
Dataset: {dataset}
Feature Extractor: {feature.upper()}
SVM Kernel: {kernel.upper()} (C={c_val})
Generated from SVM VisionLab Python Studio.
"""

import numpy as np
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split{", GridSearchCV" if grid_search else ""}
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import joblib

def load_data():
    print("[1/4] Loading {dataset} dataset...")
    digits = datasets.load_digits()
    images = digits.images
    labels = digits.target
    return images, labels

def extract_features(images):
    print("[2/4] Extracting {feature.upper()} descriptors...")
    features = []
    for img in images:
        # Resize to standardized canonical dimensions
        img_resized = cv2.resize(img, (32, 32))
        
        # Extract HOG Descriptor
        hog_desc = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        features.append(hog_desc)
        
    return np.array(features, dtype=np.float32)

def train_and_evaluate():
    images, labels = load_data()
    X = extract_features(images)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[3/4] Training SVM ({kernel.upper()} kernel) on {{X_train.shape[0]}} samples (Dim: {{X.shape[1]}})...")
    
'''
    if grid_search:
        code_template += '''    param_grid = {
        'C': [0.1, 1, 10, 50],
        'gamma': ['scale', 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear']
    }
    svm_model = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, verbose=1)
    svm_model.fit(X_train, y_train)
    best_clf = svm_model.best_estimator_
    print(f"Optimal Hyperparameters: {svm_model.best_params_}")
'''
    else:
        code_template += f'''    best_clf = SVC(kernel='{kernel}', C={c_val}, gamma='scale', probability=True, random_state=42)
    best_clf.fit(X_train, y_train)
'''

    code_template += '''
    print("[4/4] Evaluating Model Performance...")
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\\n>>> Overall Test Accuracy: {acc * 100:.2f}%\\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained model to disk
    joblib.dump(best_clf, "svm_object_model.pkl")
    print("Trained model serialized to 'svm_object_model.pkl'")

if __name__ == "__main__":
    train_and_evaluate()
'''
    return jsonify({"status": "success", "code": code_template})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" SVM Object Recognition Web Server is running!")
    print(f" Open http://localhost:{port} in your web browser.")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
