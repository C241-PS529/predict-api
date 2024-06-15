import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'lungxcan-162c632a64f0.json'
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
storage_client = storage.Client()

def custom_metric(y_true, y_pred):
    pass

model = load_model('best_model.h5', custom_objects={'custom_metric': custom_metric})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'message': 'No file part in the request'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'message': 'No file selected for uploading'}), 400

            image_bucket = storage_client.get_bucket('lungxcan-bucket')
            filename = secure_filename(file.filename)  # Ensure a secure filename
            blob = image_bucket.blob(f'predict_uploads/{filename}')
            blob.upload_from_file(file)

            img_path = BytesIO(blob.download_as_bytes())
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            images = np.vstack([x])

            # Model prediction
            pred_disease = model.predict(images)
            max_prob = pred_disease.max()

            print(f"Prediction probabilities: {pred_disease}")
            print(f"Maximum probability: {max_prob}")

            diseases = [
                'CANCER',
                'COVID',
                'FIBROSIS',
                'NORMAL',
                'PLEURAL THICKENING',
                'PNEUMONIA',
                'TBC'
            ]
            details = [
                'Hasil pemeriksaan menunjukkan adanya tanda-tanda kanker paru-paru, suatu kondisi di mana sel-sel tidak terkendali tumbuh secara abnormal di paru-paru. Gejala kanker paru-paru dapat meliputi batuk kronis, sesak napas, nyeri dada, dan kehilangan berat badan yang tidak diinginkan. Untuk penanganan yang optimal, penting untuk segera berkonsultasi dengan dokter untuk evaluasi lebih lanjut, diagnosis yang akurat, dan perencanaan pengobatan yang tepat, yang bisa mencakup terapi seperti operasi, kemoterapi, radioterapi, atau terapi target.',
                'Hasil pemeriksaan menunjukkan tanda-tanda COVID-19, penyakit yang disebabkan oleh virus Corona jenis baru, SARS-CoV-2. Gejala COVID-19 meliputi demam, batuk kering, kelelahan, dan kesulitan bernapas, dengan gejala lain seperti hilangnya penciuman atau rasa juga bisa terjadi. Untuk penanganan yang tepat, penting untuk mengisolasi diri dan segera menghubungi penyedia layanan kesehatan untuk petunjuk lebih lanjut, terutama jika gejala menjadi parah atau jika ada riwayat kontak dengan kasus terkonfirmasi COVID-19.',
                'Hasil pemeriksaan menunjukkan tanda-tanda fibrosis paru-paru, suatu kondisi di mana jaringan parut menggantikan jaringan normal di paru-paru. Fibrosis mengganggu fungsi normal paru-paru karena jaringan parut yang tidak elastis menggantikan jaringan normal yang fleksibel. Untuk diagnosis yang akurat dan perencanaan pengobatan yang optimal, penting untuk segera berkonsultasi dengan dokter untuk mendapatkan pengobatan yang tepat.',
                'Hasil pemeriksaan menunjukkan bahwa paru-paru Anda dalam kondisi normal. Ini menandakan fungsi paru-paru yang baik tanpa tanda-tanda penyakit atau kelainan. Paru-paru normal memiliki struktur yang baik dan fungsi yang optimal untuk melakukan pertukaran gas yang diperlukan untuk bernapas. Tetap menjaga gaya hidup sehat, termasuk kegiatan fisik yang teratur dan menghindari paparan zat-zat berbahaya seperti asap rokok, untuk memastikan kesehatan paru-paru yang optimal di masa depan.',
                'Hasil pemeriksaan menunjukkan adanya penebalan pleura, lapisan tipis jaringan yang melapisi paru-paru dan rongga dada. Pleural thickening bisa terjadi sebagai respons terhadap berbagai penyebab, termasuk infeksi, trauma, atau paparan bahan berbahaya seperti asbes. Penting untuk segera berkonsultasi dengan dokter untuk evaluasi lebih lanjut.',
                'Hasil pemeriksaan menunjukkan tanda-tanda pneumonia, suatu infeksi pada paru-paru yang biasanya disebabkan oleh bakteri, virus, atau jamur. Gejala pneumonia meliputi demam, batuk berdahak, kesulitan bernapas, dan nyeri dada. Penting untuk segera berkonsultasi dengan dokter untuk evaluasi lebih lanjut dan pengobatan yang tepat, yang mungkin melibatkan antibiotik, istirahat, dan perawatan suportif lainnya.',
                'Hasil pemeriksaan menunjukkan tanda-tanda tuberkulosis, suatu penyakit infeksi menular yang disebabkan oleh bakteri Mycobacterium tuberculosis. Gejala TBC meliputi batuk kronis, demam, penurunan berat badan, dan keringat malam. Untuk diagnosis dan pengobatan yang tepat, segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.'
            ]

            threshold = 0.75

            if max_prob <= threshold:
                return jsonify({'message': 'Disease not detected'}), 400

            detected_disease = diseases[np.argmax(pred_disease)]
            detected_detail = details[np.argmax(pred_disease)]
            result = {
                'diseases': detected_disease,
                'details': detected_detail
            }
            return jsonify(result), 200

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'message': f'Error processing request: {e}'}), 500

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)
