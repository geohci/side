import os
import yaml

from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)

__dir__ = os.path.dirname(__file__)
app.config.update(
    yaml.safe_load(open(os.path.join(__dir__, 'default_config.yaml'))))
try:
    app.config.update(
        yaml.safe_load(open(os.path.join(__dir__, 'config.yaml'))))
except IOError:
    # It is ok if there is no local config file
    pass

# Enable CORS for API endpoints
#CORS(app, resources={'*': {'origins': '*'}})
CORS(app)

@app.route('/')
def index():
    title = validate_api_args()
    return render_template('index.html', title=title)

def validate_api_args():
    title = None
    if 'title' in request.args:
        title = request.args['title'].replace('_', ' ')

    return title