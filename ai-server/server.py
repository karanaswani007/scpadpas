import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

firebase_json = os.environ.get("FIREBASE_KEY")

firebase_dict = json.loads(firebase_json)

cred = credentials.Certificate(firebase_dict)

firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="animal_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = image.resize((224,224))

    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0][0]

    if output > 0.70:
        result = "Wild"

    elif output < 0.0005:
        result = "NonWild"

    else:
        result = "NoAnimal"

    db.collection("detections").add({
    "device_id": "FieldCam01",
    "prediction": result,
    "confidence": float(output),
    "timestamp": datetime.now()
    })

    return jsonify({
        "prediction": result,
        "confidence": float(output)
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)



