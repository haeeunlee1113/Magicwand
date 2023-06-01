from flask import app
import requests
import urllib3
from model.editor.video import Video
from model.editor.editor import MagicWand
from flask import Flask

app = Flask(__name__)

magicwand = MagicWand()
raw_path = "videos/samples.mp4"
Video = Video(raw_path)


@app.route('/')
def hello_world(video=Video):
    # remove_video = magicwand.remove_silence(video)
    stt_video = magicwand.run_stt(video)
    emotion = magicwand.run_ser(video)

    return 'upload end!'


'''@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name) '''

if __name__ == '__main__':
    app.run()
