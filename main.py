from flask import Flask, request, jsonify
from flask_cors import CORS
from classify_diseases import predict_disease
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No any Image fiiles in your request."}), 400
    file1 = int(request.args.get('file'))
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Proide a valid file."}), 400
    image_stream = file.read()
    image = Image.open(BytesIO(image_stream)).convert("RGB")
    image = np.array(image)
    try:
        disease, prediction, features, processed_urls = predict_disease(image, file1)
        return jsonify(
            {
                "features": features.tolist(),
                "disease": disease,
                "prediction": prediction,
                "urls": processed_urls,
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
