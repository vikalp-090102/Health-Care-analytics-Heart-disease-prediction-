from flask import Flask, render_template, request,  url_for
app = Flask(_name_)

@app.route('/')
def hello_world():
    return render_template('test.html')

@app.route('/HeartDisease.html')
def hello_world2():
    return render_template('heartdisease.html')

@app.route('/predict', methods = ["POST"])
def predict():
    row = []
    if request.method == 'POST':
        row.append(request.form['age'])
        row.append(request.form['Gender'])
        row.append(request.form['cp'])
        row.append(request.form['trestbps'])
        row.append(request.form['chol'])
        row.append(request.form['fbs'])
        row.append(request.form['restecg'])
        row.append(request.form['thalach'])
        row.append(request.form['exang'])
        row.append(request.form['oldpeak'])
        row.append(request.form['slope'])
        row.append(request.form['ca'])
        row.append(request.form['thal'])
        
        import numpy as np
        row = np.array(row)
        row = row.astype(float)
        import randomforest2
        a1 = randomforest2.findAns(row)
        import knn
        a2 = knn.findAns2(row)
        return render_template('result.html', ans1=a1, ans2 = 0, ans3 = 0)
    

if _name_ == "_main_":
    app.run(debug=True)
