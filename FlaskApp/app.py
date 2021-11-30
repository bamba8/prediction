import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("mod.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [int(float(x)) for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = final_features.reshape(1, -1)
    prediction = model.predict(final_features)
    prediction_text = ""
    if prediction == 1:
        prediction_text = "Vous avez un probleme cardiaque ."
    else:
        prediction_text = "Vous etes bien portant"
    return render_template('index.html', predict=prediction, prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)