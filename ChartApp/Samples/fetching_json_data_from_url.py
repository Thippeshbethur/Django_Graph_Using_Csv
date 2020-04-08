from django.shortcuts import render
from django.http import HttpResponse
from ChartApp import views

# Include the `fusioncharts.py` file which has required functions to embed the charts in html page
from ChartApp.fusioncharts import FusionCharts

# Loading Data from a Static JSON String
# It is a example to show a Column 2D chart where data is passed as JSON string format.
# The `chart` method is defined to load chart data from an JSON string.

def chart(request):
  # Create an object for the column2d chart using the FusionCharts class constructor
  filename=request.FILES["excel_file"]
  context=views.getdata(request)  
    # returning complete JavaScript and HTML code, which is used to generate chart in the browsers.
  return  render(request, 'catalogue.html',context)
