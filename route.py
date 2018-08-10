from flask import Flask, render_template, request, flash, redirect, url_for
from flask_pymongo import PyMongo
from ast import literal_eval
from bson.json_util import dumps
import json
import os
from bson import ObjectId
from werkzeug.utils import secure_filename
import numpy as np 
from sklearn.cluster import KMeans

# Configs
np.set_printoptions(threshold=np.inf)
UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/facial"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
mongo = PyMongo(app)

# Calculates Euclidian Distance between two embeddings.
def distance(embed1, embed2):
    dist = np.sqrt(np.sum(np.square(np.subtract(embed1, embed2))))
    return dist

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
  data = list(mongo.db.index.find({}))
  for item in data:
    item['_id'] = str(item['_id'])
  return render_template('index.html', mongo_data=data)

@app.route('/convert', methods=['GET'])
def convert():
  return render_template('convert.html')

@app.route('/info', methods=['GET'])
def info():
  return render_template('info.html')

@app.route('/save-embedding', methods=['POST'])
def save():
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    data = json.loads(request.form['data'])
    data['path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mongo.db.index.insert_one(data)
    return JSONEncoder().encode(data)

@app.route('/compare', methods=['POST'])
def compare():
  if request.method == 'POST':
    face = literal_eval((request.data).decode('utf8'))['face']
    embeddings = list(mongo.db.index.find({}))
    # data must take in embedding and array of embedding
    matches = []
    for embed in embeddings:
      print(distance(face, embed['embedding']))
      if (distance(face, embed['embedding']) < 1.1):
          matches.append(embed)
    return dumps({ 'matches': matches })

class JSONEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, ObjectId):
      return str(o)
    return json.JSONEncoder.default(self, o)