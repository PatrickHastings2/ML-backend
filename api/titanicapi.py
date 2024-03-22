# Import necessary stuff
from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.titanic import dt, logreg, enc, cols

# Initialize Flask
app = Flask(__name__)

# Create Titanic API
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')

# Predict survival using a POST request
@titanic_api.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json(force=True)
        features = pd.DataFrame(data, index=[0])

        # Convert data for the model
        onehot = enc.transform(features[['embarked']]).toarray()
        features = features.join(pd.DataFrame(onehot, columns=cols))
        features.drop(['embarked'], axis=1, inplace=True)

        # Make sure there's no missing data
        # Replace missing values with some defaults if needed
        # Example: features.fillna(0, inplace=True)

        # Predict with models and get survival probability
        prediction_dt_proba = dt.predict_proba(features)
        prediction_logreg_proba = logreg.predict_proba(features)

        survival_probability_dt = prediction_dt_proba[0][1]
        survival_probability_logreg = prediction_logreg_proba[0][1]

        # Return predictions as percentages
        return jsonify({
            'DT Probability': f"{survival_probability_dt:.2%}",
            'LogReg Probability': f"{survival_probability_logreg:.2%}"
        })
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)})

# Register the API with the main app
# app.register_blueprint(titanic_api)

if __name__ == '__main__':
    app.run(debug=True)  # Run with debug mode
