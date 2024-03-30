import numpy as np
from flask import Flask,render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))



@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def piford_salary_predictor():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Employee Salary should be RS {}'.format(output))







if __name__ == "__main__":
    app.run(debug=True)