import os
from django.shortcuts import render
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt 
import base64
from io import BytesIO

import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



# it is a default view.
# please go to the samples folder for others view

Json_chart = {}
Json_chart['chart'] = []
Json_chart['data']=[]

def catalogue(request):
    return render(request, 'catalogue.html')


def getdata(request):
    dataSource = {}
    excel_file = request.FILES['excel_file']
    data = pd.read_csv(excel_file)
    data.shape
    X = data['age'].values.reshape(-1,1)
    y = data['health'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    if(request.POST["DrpValue"]=="Linear Regression"):
      regressor = LinearRegression()  
      regressor.fit(X_train, y_train) #training the algorithm
      y_pred = regressor.predict(X_test)
    elif  request.POST["DrpValue"]=="Logistic Regression":
      regressor = LogisticRegression()  
      regressor.fit(X_train, y_train) #training the algorithm
      y_pred = regressor.predict(X_test)
    elif request.POST["DrpValue"]=="Decision Tree":
      regressor = DecisionTreeClassifier()  
      regressor.fit(X_train, y_train) #training the algorithm
      y_pred = regressor.predict(X_test)
    df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df2 = df1.head(15)
    df2.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')      
    plt.savefig("ChartApp/Templates/static/"+request.POST["DrpValue"])  

    context= {
      'person': request.POST["DrpValue"]
    } 
    return context

    # plt.show()
    # Data_frame=pd.DataFrame(data)

    # gk = Data_frame.groupby(["sex"], as_index=False)["Student_id"].count()
    # df = json.loads(data.to_json(orient='table'))
    
    # dataSource['chart']: {
    #      "caption": "Number of male and female users",
    #      "subCaption": "Number of male and female users",
    #      "xAxisName": "Gender",
    #      "yAxisName": "Marks in test1",
    #      "exportEnabled": "1",
    #      "theme": "fusion",
    #      "width":10
    # }

    # dataSource['data'] = []
    # for key in gk.values:
    #   data = {}
    #   data['label'] = key[0]
    #   data['value'] = key[1]
    #   dataSource['data'].append(data)  
    
    # return dataSource
