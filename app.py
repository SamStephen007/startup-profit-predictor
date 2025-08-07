from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the Ridge model and feature names
model = pickle.load(open("model/ridge_model.pkl", "rb"))
feature_names = pickle.load(open("model/features.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        rd = float(request.form['rd'])
        admin = float(request.form['admin'])
        marketing = float(request.form['marketing'])
        state = request.form['state']  # State: 'California', 'Florida', 'New York'

        # Initialize input feature dictionary
        input_dict = {
            'R&D Spend': rd,
            'Administration': admin,
            'Marketing Spend': marketing,
            'State_Florida': 0,
            'State_New York': 0
        }

        # Manually encode the state (drop_first=True => California is the baseline)
        if state == 'Florida':
            input_dict['State_Florida'] = 1
        elif state == 'New York':
            input_dict['State_New York'] = 1
        # California => both are 0

        # Arrange the features in the correct order
        input_data = [input_dict.get(col, 0) for col in feature_names]

        # Convert to NumPy array and predict
        input_array = np.array(input_data).reshape(1, -1)
        predicted_profit = model.predict(input_array)[0]

        return render_template('index.html', prediction_text=f"Predicted Profit: ${predicted_profit:,.2f}")

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
