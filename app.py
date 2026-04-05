from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prediction = model.predict([[hours]])
    result = f"Predicted Score: {prediction[0]:.2f}"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)