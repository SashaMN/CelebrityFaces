from flask import Flask, render_template, request
import os
import re
import io
import base64
from PIL import Image
app = Flask(__name__)
import requests
from subprocess import run, PIPE
import json
import sys
import numpy as np


@app.route('/')
def upload():
    return render_template('upload.html')

filenames = sorted(list(os.listdir('data/img_align_celeba/')))
embeddings = np.load('embeddings.npy')

def RBF(x, gamma=1.0):
    return np.exp(-x * gamma)

@app.route('/similarities', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        oldF = request.files['file']
        path = os.path.join('static/data', oldF.filename)
        print(path, file=sys.stderr)
        imgdata = re.search(r'base64,(.*)', request.form['base64Img']).group(1) #This is a hack to clean up the encoding.
        imgdata = base64.b64decode(imgdata)
        img_handler = io.BytesIO(imgdata)
        img = Image.open(img_handler)
        img.save(path)
        return render_template('result.html', result = process(oldF.filename))

def build_output(closest_neighbors_id, values):
    scores = []
    for i in closest_neighbors_id:
        score = ((embeddings[i, :] - np.array(values)) ** 2).sum()
        scores.append(score)

    res = []
    idx = np.argsort(scores)[:5]
    for i in idx:
        cur_id = closest_neighbors_id[i]
        prob = RBF(scores[i]) * 100
        res.append(("{0:.2f}%".format(prob),
            filenames[cur_id]))
    
    return res

def process(path):
    full_path = os.path.join('static/data', path)
    response = requests.post(url="http://127.0.0.1:50001", files = {'file': open(full_path, 'rb')})
    values = json.loads(response.content)
    p = run(['bin/index_server'],
            input=' '.join(map(str, values)).encode(),
            stdout=PIPE)
    return {
        ("Our approach", path): build_output(list(map(int, p.stdout.decode().strip().split())), values),
        ("Naive approach", path): build_output(list(range(202599)), values)
    }

if __name__ == '__main__':
    app.run(host='95.213.170.235', debug=False, port=50002)
