import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'lungxcan-162c632a64f0.json'
storage_client = storage.Client()

def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req

# Load model
model = load_model('best_model.h5', custom_objects={'req': req})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get the bucket containing the image
            image_bucket = storage_client.get_bucket('lungxcan-bucket')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict_uploads/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        try:
            # Load and preprocess the image
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            images = np.vstack([x])

            # Model prediction
            pred_animal = model.predict(images)
            maxx = pred_animal.max()

            # Labels and information
            

            # Check if the prediction confidence is above threshold
            if maxx <= 0.75:
                respond = jsonify({
                    'message': 'Penyakit tidak terdeteksi'
                })
                respond.status_code = 400
                return respond

            # Prepare the result
            result = {
                
            }

            respond = jsonify(result)
            respond.status_code = 200
            return respond
        except Exception as e:
            respond = jsonify({'message': str(e)})
            respond.status_code = 500
            return respond

    return 'OK'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
