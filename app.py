import json
import pickle

from flask import Flask,request,app,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
rf_model=pickle.load(open('best_rf.pkl','rb'))
label_encoder_sec_name=pickle.load(open('label_encoder_sec_name.pkl','rb'))
label_encoder_state_des=pickle.load(open('label_encoder_state_des.pkl','rb'))


@app.route('/')
def home(): 
    #print("it has started")
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    data[0]=int(data[0])
    data[1]=int(data[1])
    data[2]=label_encoder_state_des.transform(np.array(data[2]).reshape(1,-1))[0]
    data[3]=label_encoder_sec_name.transform(np.array(data[3]).reshape(1,-1))[0]

    print(data)
    output=rf_model.predict(np.array(data).reshape(1,-1))
    return render_template("home.html",prediction_text="The price prediction is {}".format(output))





if __name__=="__main__":
    app.run(debug=True)
   
     