from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    data1 = request.form['sepal_length']
    data2 = request.form['sepal_width']
    data3 = request.form['petal_length']
    data4 = request.form['petal_width']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('index.html', data=pred)


if __name__ == "__main__":
   app.run()