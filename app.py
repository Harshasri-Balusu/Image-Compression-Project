from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    # Get uploaded image
    img = Image.open(request.files['file'].stream).convert('L')
    img_arr = np.array(img)

    # Apply SVD on image
    u, s, vh = np.linalg.svd(img_arr, full_matrices=False)

    # Get user specified rank and compress image
    rank = int(request.form['rank'])
    img_compressed_arr = np.dot(u[:, :rank], np.dot(np.diag(s[:rank]), vh[:rank, :]))
    img_compressed = Image.fromarray(np.uint8(img_compressed_arr))

    # Save compressed image to byte buffer
    byte_io = io.BytesIO()
    img_compressed.save(byte_io, 'JPEG', quality=95)
    byte_io.seek(0)

    return send_file(byte_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
