from flask import app
import requests
import urllib3
from model.editor.editor import MagicWand
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
from model.editor.video import Video
UPLOAD_FOLDER = 'videos'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'mp4'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
magicwand = MagicWand()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



@app.route('/encoding/<name>')
def hello_world(name):
    print(UPLOAD_FOLDER + "/" + name)
    video = Video(UPLOAD_FOLDER + "/" + name)
    # remove_video = magicwand.remove_silence(video)
    filename = video.video_path
    stt_video = magicwand.run_stt(video)
    emotion = magicwand.run_ser(video)
    magicwand.encode_subtitle(video, emotion)
    return "./videos/encoded_video.mp4"


@app.route('/uploads/<name>')
def download_file(name):
    send_from_directory(app.config["UPLOAD_FOLDER"], name)

    return redirect(url_for('hello_world', name=name))



if __name__ == '__main__':
    app.run()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
