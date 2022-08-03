import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pandas as pd
import joblib
import csv
from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence

app = Flask(__name__)
model_PseAAC5 = joblib.load("Model_SVM_PseAAC5")
model_PseAAC10= joblib.load("Model_SVM_PseAAC10")
def check_seq(sequence):
    AALetter=   ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V",
                 "a","r","n","d","c","e","q","g","h","i","l","k","m","f","p","s","t","w","y","v"]
    counter= 0
    for k in sequence:
        counter = counter+1
        if k not in AALetter: 
            return "invalid"
    if counter < 6:
        return "invalid"
    return "valid"
        
def model_predict(sequence):
    AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    sequence= sequence.upper()
    final_features = []
    counter=0
    for j in sequence:
        counter = counter+1
  
    if counter > 5 and counter <= 10:
        DesObject = PyPro.GetProDes(sequence)      
        paac = DesObject.GetPAAC(lamda=5,weight=0.05)
        k=1
        while k<26:
            final_features.append(paac['PAAC'+str(k)])
            k+=1
        final_features=np.array(final_features)
        return model_PseAAC5.predict(final_features.reshape(1,-1))
    elif counter > 10:
        DesObject = PyPro.GetProDes(sequence)      
        paac = DesObject.GetPAAC(lamda=10,weight=0.05)
        k=1
        while k<31:
            final_features.append(paac['PAAC'+str(k)])
            k+=1
        final_features=np.array(final_features)
        return model_PseAAC10.predict(final_features.reshape(1,-1))
    
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    sequence = request.form.get("sequence")
    if check_seq(sequence)=="invalid":
        return render_template('index.html', return_text = "Heads Up!! Enter valid sequence. Please return and reload the page")
    elif model_predict(sequence)==0:
        final= 'Not a Biofilm inhibiting peptide'
    elif model_predict(sequence)==1:
        final='It is a Biofilm inhibiting peptide'
        
    return render_template('index.html', return_text=final)                          


if __name__ == "__main__":
    app.run(debug=False)