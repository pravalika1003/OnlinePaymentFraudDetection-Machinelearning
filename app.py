from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('dct_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    type = float(request.form['type'])
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])

    # Make prediction
    features = np.array([[type, amount, oldbalanceOrg, newbalanceOrig]])
    prediction = model.predict(features)

    # Translate prediction to human-readable text
    #fraud_status = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

    return render_template('index.html', prediction_text=f"The transaction is predicted to be {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
