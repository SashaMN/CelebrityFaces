import numpy as np
import tensorflow as tf
import requests
import os
import imageio
from skimage.transform import resize
import sys
from flask import Flask, request, jsonify

app = Flask(__name__)

META_FILE = "model/20170512-110547/model-20170512-110547.meta"
CKPT_FILE = "model/20170512-110547/model-20170512-110547.ckpt-250000"

def load_model(meta_file, ckpt_file):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, ckpt_file)
    return sess

sess = load_model(META_FILE, CKPT_FILE)
graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")
phase_train = graph.get_tensor_by_name("phase_train:0")

OUTPUT_SHAPE = (160, 160)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['file']
        image = imageio.imread(image)
        image = resize(image, output_shape=OUTPUT_SHAPE, mode='reflect')[None, :, :, :]
        img_emb = sess.run(embeddings, feed_dict={
            input_tensor : image,
            phase_train : False})[0]
        return jsonify(img_emb.tolist())

app.run(port=50001)
