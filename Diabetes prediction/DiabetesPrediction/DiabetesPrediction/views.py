from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from django import forms
model = pickle.load(open('model.pkl', 'rb'))


dataset = pd.read_csv('d:diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


def home(request):
    return render(request,'home.html')

def predict(request):
    '''
    For rendering results on HTML GUI
    '''
    all=request.POST
    data={}
    for i,j in all.items():
        data.update({i:j})
    data.pop('csrfmiddlewaretoken')
    float_features = [float(x) for x in data.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )



    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render(request,'predict.html', {'prediction_text':'{}'.format(output)})
