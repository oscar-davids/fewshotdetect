
import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, Response, jsonify, make_response
from werkzeug.utils import secure_filename
from trainmodel import trainmodel
from loadmodel import loadmodel, objectdetect

app=Flask(__name__,template_folder='templates')

IMAGE_PROCESS_OK = 100
IMAGE_PROCESS_ERR = 101
INVALID_REQUEST_ERR = 231

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'traindata')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'csv'])
# trained model
MODEL_FOLDER = os.path.join(path, 'models')

MODEL_NAME = ""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getmodels():
    models = []
    for (_, _, file) in os.walk(MODEL_FOLDER):
        for f in file:
            if '.index' in f:
                models.append(f.replace(".index", ""))
    return models


@app.route('/')
def upload_form():
    models = getmodels()
    return render_template('upload.html', models = models)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        MODEL_NAME = request.form['model']

        if 'files[]' not in request.files:
            flash('No Image Files')
            return redirect(request.url)

        if 'label' not in request.files:
            flash('No Label File')
            return redirect(request.url)

        modelpath = os.path.join(app.config['UPLOAD_FOLDER'], MODEL_NAME)        

        if not os.path.isdir(modelpath):
            os.mkdir(modelpath)

        imgfiles = request.files.getlist('files[]')

        for file in imgfiles:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(modelpath, filename))

        csvfile = request.files.getlist('label')[0]
        if csvfile and allowed_file(csvfile.filename):
                filename = secure_filename(csvfile.filename)
                csvfile.save(os.path.join(modelpath, "label.csv"))


        flash('File(s) successfully uploaded')
        trainmodel(MODEL_NAME)

        return redirect('/')

@app.route('/loadmodel/', methods=['POST'])
def loadmodel():
    if request.method == 'POST':
        model = str(request.form.get('comp_select'))
        loadmodel(model)
    #deploy other host and load sample

    return redirect('/')

@app.route('/objectdetection', methods=['GET', 'POST'])
def objectdetection():
    if request.method == 'GET':
        return Response('objectdetection server is running.', status=200)

    # POST
    # Read image data
    img_data = request.json
    if 'image' not in img_data:
        response = {
            'error': IMAGE_PROCESS_ERR
        }
        return make_response(jsonify(response), 400)

    candidates = objectdetect(img_data)
    return make_response(jsonify(candidates), 200)

@app.route('/download/<filename>',methods=['POST'])
def download(filename):    
    return send_from_directory(MODEL_FOLDER, filename) 
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5555,debug=False,threaded=True)