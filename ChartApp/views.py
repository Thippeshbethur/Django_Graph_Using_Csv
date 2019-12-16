from django.shortcuts import render
import pandas as pd
import csv
import json

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
    Data_frame=pd.DataFrame(data)
    gk = Data_frame.groupby(["sex"], as_index=False)["Student_id"].count()
    df = json.loads(data.to_json(orient='table'))
    
    dataSource['chart']: {
        "caption": "Number of male and female users",
        "subCaption": "Number of male and female users",
        "xAxisName": "Gender",
        "yAxisName": "Marks in test1",
        "exportEnabled": "1",
        "theme": "fusion",
        "width":10
    }

    dataSource['data'] = []
    for key in gk.values:
      data = {}
      data['label'] = key[0]
      data['value'] = key[1]
      dataSource['data'].append(data)  
    
    return dataSource
