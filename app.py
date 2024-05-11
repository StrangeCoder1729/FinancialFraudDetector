from flask import Flask, request, render_template
import numpy as np
import pickle
from DataPreProcessing  import sc
# from model1 import classifier
app = Flask(__name__)

# Load the models
models = {}
models['logisticModel'] = pickle.load(open('logisticModel.pkl', 'rb'))
models['decisionModel'] = pickle.load(open('decisionModel.pkl', 'rb'))
models['randomForestmodel'] = pickle.load(open('randomForestmodel.pkl', 'rb'))
models['xgBoostModel'] = pickle.load(open('XGBoostModel.pkl','rb'))

# Load the scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form inputs
    amount = float(request.form['amount'])
    old_balance = float(request.form['old_balance'])
    new_balance = float(request.form['new_balance'])
    transaction_category = int(request.form['transaction_category'])

    # Prepare input features for prediction
    input_features = np.array([[transaction_category,amount, old_balance, new_balance]])

    # Scale the input features
    input_features_scaled = sc.transform(input_features)

    # Perform predictions for the selected model
    selected_model_name = request.form['type']
    selected_model = models[selected_model_name]

    if selected_model_name in ['randomForestmodel', 'decisionModel']:
        # Predict class label directly
        prediction = selected_model.predict(input_features_scaled)[0]
    else:  # For logisticModel or other models
        # Predict probability of fraud
        prediction = selected_model.predict_proba(input_features)[0][1]

    # Determine the result based on prediction
    if prediction >= 0.5:
        result = f'{selected_model_name}: Fraud Detected'
    else:
        result = f'{selected_model_name}: No Fraud'

    return render_template('index.html', pred=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
