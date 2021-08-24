import os
import codecs
import datetime
import cv2
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from matplotlib import pyplot as plt
import base64
import io
import xml.etree.ElementTree as ET
import numpy as np
import pdfkit
import dash_table
import plotly.express as px
import plotly.graph_objects as go
os.remove('Test_Report.csv')
col=['TB','Covid','PE']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {'background': '#111111'}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
##x1=['TB','Covid','PE']
##y4=[0,0,0]
##fig = go.Figure(data=[go.bar(x=x1, y = y4)])
app.layout = html.Div([
                # adding a header and a paragraph
                html.Div([
                    html.H1("Acculi CHEST X-ray SCREENING SYSTEM."),
                    html.P("R.R. Nagar,Banglore,560098,Karnataka,India"),
                         ], 
                    style = {'padding' : '50px' , 
                             'backgroundColor' : 'lightblue',
                             'textAlign': 'center'}),
    html.Div(id='output-image-upload',
                style={'backgroundColor' : 'lightgreen','textAlign':'left'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop',
            #html.A('Select Files')
        ]),
        style={
            'width': '10%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '5px',
            'backgroundColor': 'lightblue',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
     html.Div([
     html.Button('Classify', id='btn-nclicks-1', n_clicks=0),
     html.Button('Show Process Data', id='btn-nclicks-2', n_clicks=0)]),           
     #style={id:'btn-nclicks-2','backgroundColor' : 'lightgreen','textAlign':'center','width': '10%','height': '60px','lineHeight': '60px','borderWidth': '1px',
                                                                   #'borderRadius': '5px','margin': '5px','borderStyle': 'dashed'}),
     html.Div(id='container-button-timestamp'),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in col]),
    html.Hr(),
    dcc.Graph(
        id='plot'),         
    html.P('Doctor Validation Result'),
                 dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'TB Covid PE', 'value': 'TB'},
            {'label': 'Covid TB PE', 'value': 'Covid'},
            {'label': 'PE Covid TB', 'value': 'PE'}
        ],
    ),
    html.Div(id='dd-output-container'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = content_string.encode("utf-8")
    s = decoded.decode('UTF-8')
    print(decoded)
    #base64_string = decoded.encode("UTF-8")
    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # save image data
        html.Img(src=contents),
        html.H5('Test X-ray Image(TB,Covid-19,Pleural effusion'),
        html.Hr(),
        
        
##       html.Pre(contents[0:200] + '...', style={
##            'whiteSpace': 'pre-wrap',
##           'wordBreak': 'break-all'
 ##      })
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    a=list_of_contents
    if list_of_contents is not None:
        list1=[list_of_names]
        list2=['Imagename']
        df=pd.DataFrame({'Imagename' : list1[0]},columns=['Imagename'])
        df.to_csv('TestImageName.csv',index=None)
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'))
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = 'Process is Going on'
        df=pd.read_csv('TestImageName.csv')
        list1= df['Imagename'].tolist()
        print(list1)
        img = cv2.imread(list1[0],cv2.IMREAD_COLOR)
        h, w, c = img.shape
        y=int(h*16/100)
        x=int(w*16/100)
        crop_image = img[x:w,y:h]
        grayimg = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        Gaussian_Iamge = cv2.GaussianBlur(grayimg,(5,5),0)
        edges = cv2.Canny(Gaussian_Iamge,30,30)
        df = pd.DataFrame(edges/255)
        Descibe_Data=df.describe()
        df1=Descibe_Data.mean(axis=1)
        med1=df1.iloc[1:len(df1)].mean(axis=0)
        df2=Descibe_Data.median(axis=1)
        med2=df2.iloc[1:len(df2)].mean(axis=0)
        df3=Descibe_Data.std(axis=1)
        med3=df3.iloc[1:len(df3)].mean(axis=0)
        df4=pd.read_csv('TB.csv')
        df5=pd.read_csv('Covid.csv')
        df6=pd.read_csv('PE.csv')
        data_point_TB=np.array([df4['Mean'],df4['Median']])
        data_point_Covid=np.array([df5['Mean'],df5['Median']])
        data_point_PE=np.array([df6['Mean'],df6['Median']])
        daat_point_test_image=np.array([round(med1,4),round(med2,4)])
        Euclidean_distance_TB = round(np.linalg.norm(data_point_TB - daat_point_test_image),4)
        Euclidean_distance_Covid = round(np.linalg.norm(data_point_Covid - daat_point_test_image),4)
        Euclidean_distance_PE = round(np.linalg.norm(data_point_PE - daat_point_test_image),4)
        list9=[str("TB"),str("Covid"),str("PE")]
        list8=[Euclidean_distance_TB, Euclidean_distance_Covid,Euclidean_distance_PE]
        new_data=pd.DataFrame({'TB' : [list8[0]],
                                   'Covid' : [list8[1]],
                                         "PE" : [list8[2]]}, 
                                  columns=['TB', 'Covid','PE'])
        new_data.to_csv('Test_Report.csv',index=None)
        msg='Process is Complete.click on "Show Process Data"'
    else:
        msg = ''
    return html.Div(msg)
@app.callback(Output('table', 'data'),
              Input('btn-nclicks-2', 'n_clicks')
)
def displayClick(btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-2' in changed_id:
        df=pd.read_csv('Test_Report.csv')
        data=df.to_dict('records')
        return data
@app.callback(Output('plot', 'figure'),
              Input('btn-nclicks-2', 'n_clicks')
)
def updates_charts(btn2):
    df=pd.read_csv('Test_Report.csv')
    y1=df['TB'].to_list()
    y2=df['Covid'].to_list()
    y3=df['PE'].to_list()
    x1=['TB','Covid','PE']
    y4=[*y1,*y2,*y3]
    fig = go.Figure([go.Bar(x=x1, y=y4,text=y4,textposition='auto')])
    return fig
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'According to doctor have Seletced result is "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=False)
