import os
from django.shortcuts import render
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt 
import base64
from io import BytesIO

import seaborn as sns
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
    df_train = pd.read_csv(excel_file)
    df_train.shape
    df_train['Gender'] = df_train['Gender'].fillna( df_train['Gender'].dropna().mode().values[0] )
    df_train['Married'] = df_train['Married'].fillna(df_train['Married'].dropna().mode().values[0] )
    df_train['Dependents'] = df_train['Dependents'].fillna( df_train['Dependents'].dropna().mode().values[0] )
    df_train['Self_Employed'] = df_train['Self_Employed'].fillna(df_train['Self_Employed'].dropna().mode().values[0] )
    df_train['LoanAmount'] = df_train['LoanAmount'].fillna( df_train['LoanAmount'].dropna().mean() )
    df_train['Loan_Amount_Term'] = df_train['Loan_Amount_Term'].fillna( df_train['Loan_Amount_Term'].dropna().mode().values[0] )
    df_train['Credit_History'] = df_train['Credit_History'].fillna(df_train['Credit_History'].dropna().mode().values[0] )
    
    grid = sns.FacetGrid(df_train, row='Married', col='Loan_Status', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
    grid.add_legend()
    # grid = sns.FacetGrid(df_train, row='Gender', col='Loan_Status', size=2.2, aspect=1.6)
    # grid.map(plt.hist, 'ApplicantIncome', alpha=.5, bins=10)
    # grid.add_legend() 
    flg, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (14,6))

    # sns.distplot(df_train['ApplicantIncome'], ax = axes[0]).set_title('ApplicantIncome Distribution')
    # axes[0].set_ylabel('ApplicantIncomee Count')

    # sns.distplot(df_train['CoapplicantIncome'], color = "r", ax = axes[0]).set_title('CoapplicantIncome Distribution')
    # axes[0].set_ylabel('CoapplicantIncome Count')

    # sns.distplot(df_train['LoanAmount'],color = "g", ax = axes[1]).set_title('LoanAmount Distribution')
    # axes[1].set_ylabel('LoanAmount Count')

    sns.boxplot(x="Dependents",y="Loan_Status",data=df_train)
    axes[2].set_ylabel('ApplicantIncomee Count')

    plt.tight_layout()
    #plt.show()
    # X = df_train['Gender'].fillna(df_train['Gender'].dropna().mode().values[0])
    # y = df_train['LoanAmount'].values.reshape(-1,1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # if(request.POST["DrpValue"]=="Linear Regression"):
    #   regressor = LinearRegression()  
    #   regressor.fit(X_train, y_train) #training the algorithm
    #   y_pred = regressor.predict(X_test)
    # elif  request.POST["DrpValue"]=="Logistic Regression":
    #   regressor = LogisticRegression()  
    #   regressor.fit(X_train, y_train) #training the algorithm
    #   y_pred = regressor.predict(X_test)
    # elif request.POST["DrpValue"]=="Decision Tree":
    #   regressor = DecisionTreeClassifier()  
    #   regressor.fit(X_train, y_train) #training the algorithm
    #   y_pred = regressor.predict(X_test)
    # df1 = pd.DataFrame({'Actual': X_test.flatten(), 'Predicted': y_pred.flatten()})
    # df2 = df1.head(30)
    # plt.bar(df2['Actual'], df2['Predicted'])
    # plt.xlabel('Stay_In_Current_City_Years')
    # plt.ylabel('Purchase')
    # plt.title(request.POST["DrpValue"])
    # plt.legend()
    # plt.ylim(0,15000)    
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')     
    plt.savefig("ChartApp/Templates/static/"+request.POST["DrpValue"])  

    context= {
      'person': request.POST["DrpValue"]
    } 
    return context

