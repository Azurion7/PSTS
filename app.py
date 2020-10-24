from flask import Flask, jsonify, flash, request, render_template
from werkzeug.utils import secure_filename
from model import Model
from firestore_helper import FirestoreHelper
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os


app = Flask(__name__)

#UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


slots = {}
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
model = Model(MODEL_DIR, COCO_MODEL_PATH)
slots['xyz'] = pd.read_csv('./slots/xyz.csv').values
graph = tf.get_default_graph()
fh = FirestoreHelper('./xyz.json')


@app.route('/')
def home():
    return render_template('home.html')


def predict(frame, filename):
    global model, slots, graph

    name = filename.split('.')[0]

    with graph.as_default():
        n_cars, status, frame = model.get_occupancy_status(frame, slots[name])

    fh.update_status(name, status)
    fh.update_counts(name, n_cars, len(status) - n_cars)
    fout = './static/output/{}.png'.format(filename.rsplit('.', 1)[0])
    plt.imsave(fout, frame)

    response = jsonify({
        'n_cars': n_cars,
        'status': status,
        'filename': fout
    })
    return response


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({
                    'n_cars': False,
                    'status': False,
                    'filename': False
                })
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return jsonify({
                    'n_cars': False,
                    'status': False,
                    'filename': False
                })
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pil_img = Image.open(file)
            frame = np.array(pil_img)
            response = predict(frame, filename)
            return response


if __name__ == '__main__':
    app.debug = True
    app.run()
