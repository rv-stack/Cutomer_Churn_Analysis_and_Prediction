from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = Flask(__name__)

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    # Apply label encoders to categorical columns
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform([input_df[col][0]])

    # Apply scaler to numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    # Make prediction
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0][1]
    return "Churn" if prediction == 1 else "No Churn", round(probability * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            input_data = {
                'gender': request.form.get('gender'),
                'SeniorCitizen': int(request.form.get('SeniorCitizen')),
                'Partner': request.form.get('Partner'),
                'Dependents': request.form.get('Dependents'),
                'tenure': int(request.form.get('tenure')),
                'PhoneService': request.form.get('PhoneService'),
                'MultipleLines': request.form.get('MultipleLines'),
                'InternetService': request.form.get('InternetService'),
                'OnlineSecurity': request.form.get('OnlineSecurity'),
                'OnlineBackup': request.form.get('OnlineBackup'),
                'DeviceProtection': request.form.get('DeviceProtection'),
                'TechSupport': request.form.get('TechSupport'),
                'StreamingTV': request.form.get('StreamingTV'),
                'StreamingMovies': request.form.get('StreamingMovies'),
                'Contract': request.form.get('Contract'),
                'PaperlessBilling': request.form.get('PaperlessBilling'),
                'PaymentMethod': request.form.get('PaymentMethod'),
                'MonthlyCharges': float(request.form.get('MonthlyCharges')),
                'TotalCharges': float(request.form.get('TotalCharges')),
            }

            prediction, probability = make_prediction(input_data)

        except Exception as e:
            prediction = "Error"
            probability = f"Something went wrong: {e}"

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
