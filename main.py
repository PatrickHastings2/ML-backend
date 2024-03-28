import threading
from flask import Flask, render_template, request, jsonify, Blueprint
from flask.cli import AppGroup
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import *
import pandas as pd
# Import other necessary packages and modules

from __init__ import app, db, cors  # Definitions initialization

# Setup APIs
from api.covid import covid_api
from api.joke import joke_api
from api.user import user_api
from api.player import player_api
from api.carcrashapi import car_crash_api
# Database migrations
from model.users import initUsers
from model.players import initPlayers

# Setup App pages
from projects.projects import app_projects

# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# Register other URIs
app.register_blueprint(joke_api)
app.register_blueprint(covid_api)
app.register_blueprint(user_api)
app.register_blueprint(player_api)
app.register_blueprint(app_projects)
app.register_blueprint(car_crash_api)

# Car Crash API Blueprint

# @car_crash_api.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         features = pd.DataFrame(data, index=[0])
#         prediction = model.predict(features)
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)})


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/table/')
def table():
    return render_template("table.html")

@app.before_request
def before_request():
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100', 'http://127.0.0.1:4100', 'https://nighthawkcoders.github.io']:
        cors._origins = allowed_origin

custom_cli = AppGroup('custom', help='Custom commands')

@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initPlayers()

app.cli.add_command(custom_cli)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8086")
