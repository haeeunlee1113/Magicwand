from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
