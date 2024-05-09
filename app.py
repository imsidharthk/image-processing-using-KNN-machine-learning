from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

def compute_snr(original_data, compressed_data):
    signal = np.mean(original_data ** 2)
    noise = np.mean((original_data - compressed_data) ** 2)
    return 10 * np.log10(signal / noise) if noise != 0 else float('inf')

def compress_image(image_path, k=16):
    img = Image.open(image_path)
    img_data = np.array(img)
    original_shape = img_data.shape

    img_data_flattened = img_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    compressed_img_data = kmeans.cluster_centers_[kmeans.labels_]
    compressed_img_data = compressed_img_data.reshape(original_shape).astype(np.uint8)

    compressed_image_path = os.path.join(APP_ROOT, app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
    Image.fromarray(compressed_img_data).save(compressed_image_path)

    snr_compressed = compute_snr(img_data, compressed_img_data)
    return compressed_image_path, snr_compressed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            compressed_path, snr_compressed = compress_image(filepath)
            return render_template('result.html', 
                                   image_url=url_for('get_file', filename=os.path.basename(compressed_path)),
                                   snr_compressed=snr_compressed)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
