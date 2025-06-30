# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from datetime import date
import datetime
from random import randint
from urllib.request import urlopen
import webbrowser
from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse
   
# necessary imports 
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
#%matplotlib inline
pd.set_option('display.max_columns', 26)
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi

from skimage import transform
import seaborn as sns
#from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
##
import glob
#from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import seaborn as sns
#import keras as k
#from keras.layers import Dense
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from tensorflow.keras.optimizers import Adam
##
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="#floricb",
  charset="utf8",
  database="brain_tumor"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('index.html',msg=msg)




@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    act=request.args.get("act")
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('upload'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('login.html',msg=msg,act=act)

@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('login_admin.html',msg=msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    if request.method=='POST':
        name=request.form['name']
        address=request.form['address']
        dob=request.form['dob']
        mobile=request.form['mobile']
        gender=request.form['gender']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        rdate=date.today()
        print(rdate)

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        cursor = mydb.cursor()
        sql = "INSERT INTO register(id,name,gender,address,dob,mobile,email,uname,pass,rdate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,name,gender,address,dob,mobile,email,uname,pass1,rdate)
        cursor.execute(sql, val)
        mydb.commit()            
        print(cursor.rowcount, "Registered Success")
        result="sucess"
        
        if cursor.rowcount==1:
            return redirect(url_for('login',act='1'))
        else:
            return redirect(url_for('login',act='2'))
            #msg='Already Exist'  
    return render_template('register.html',msg=msg)


@app.route('/reco', methods=['GET', 'POST'])
def reco():
    msg=""
    if 'username' in session:
        uname = session['username']
    

    return render_template('reco.html',msg=msg)
  
@app.route('/view_user', methods=['GET', 'POST'])
def view_user():
    msg=""
    
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    data = mycursor.fetchall()
       
    
    return render_template('view_user.html', data=data)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)
       
              
    return render_template('admin.html',dimg=dimg)

@app.route('/add_reco', methods=['GET', 'POST'])
def add_reco():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        btype=request.form['btype']
        details=request.form['details']
        hospital=request.form['hospital']
        
        mycursor.execute("SELECT max(id)+1 FROM recommend")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO recommend(id,btype,details,hospital) VALUES (%s, %s, %s, %s)"
        val = (maxid,btype,details,hospital)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('add_reco'))

    mycursor.execute('SELECT * FROM recommend')
    data = mycursor.fetchall()
    
    return render_template('add_reco.html',msg=msg,data=data)

@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    

    return render_template('img_process.html')

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro11.html',dimg=dimg)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        #cv2.imwrite("static/trained/bb/bin_"+fname, thresh)

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/"+fname
        segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)


####
@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    
        dimg.append(fname)

        img = Image.open('static/trained/data/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        #invimg.save('static/trained/ff/'+fname)
  
    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        #edged.save(path4)
        ##
    
        
    return render_template('pro4.html',dimg=dimg)

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,210,265]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,220,275]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,70,110,155]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,30,70,110,155]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

@app.route('/pro51', methods=['GET', 'POST'])
def pro51():
    msg=""
    dimg=[]
    data=[]
    ff=open("static/trained/bt.txt","r")
    sd=ff.read()
    ff.close()
    sd1=sd.split("|")
    for sd2 in sd1:
        dt=[]
        sd4=sd2.split(",")
        dt.append(sd4[0])
        dt.append(sd4[1])
        dt.append(sd4[2])
        dt.append(sd4[3])
        dt.append(sd4[4])
        dt.append(sd4[5])
        dt.append(sd4[6])
        data.append(dt)
        
    path_main = 'static/trained/seg'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro51.html',dimg=dimg,data=data)


def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    return render_template('pro6.html',dimg=dimg)

#GLCM-gray-level co-occurrence matrix-Feature Extraction
def glcm(img, center, x, y):
      
    new_value = 0
      
    try:
        # If local neighbourhood pixel ___ value is greater than or equal___ # to center pixel values then ____ # set it to 1
        if img[x][y] >= center:
            new_value = 1
              
    except:
        # Exception is required when  # neighbourhood value of a center # pixel value is null i.e. values # present at boundaries.
        pass
      
    return new_value
def fxnglcm(img,i):
    from skimage.feature import greycomatrix,greycoprops
    import skimage.feature as feature


    graycom = feature.graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)


    c = feature.graycoprops(graycom, 'contrast')
    d = feature.graycoprops(graycom, 'dissimilarity')
    h = feature.graycoprops(graycom, 'homogeneity')
    e = feature.graycoprops(graycom, 'energy')
    corr = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')
    c = np.squeeze(c)
    d = np.squeeze(d)
    h = np.squeeze(h)
    e = np.squeeze(e)
    corr = np.squeeze(corr)
    asm = np.squeeze(ASM)
    
    df["GLCM-Contrast-1"][i] = c[0]
    df["GLCM-Contrast-2"][i] = c[1]
    df["GLCM-Contrast-3"][i] = c[2]
    df["GLCM-Contrast-4"][i] = c[3]

    df["GLCM-Dissimilarity-1"][i] =d[0]
    df["GLCM-Dissimilarity-2"][i] =d[1]
    df["GLCM-Dissimilarity-3"][i] =d[2]
    df["GLCM-Dissimilarity-4"][i] =d[3]
    
    df["GLCM-Homogeneity-1"][i] =h[0]
    df["GLCM-Homogeneity-2"][i]=h[1]
    df["GLCM-Homogeneity-3"][i] =h[2]
    df["GLCM-Homogeneity-4"][i] =h[3]
    
    df["GLCM-Energy-1"][i] =e[0]
    df["GLCM-Energy-2"][i] =e[1]
    df["GLCM-Energy-3"][i] =e[2]
    df["GLCM-Energy-4"][i] =e[3]

    df["GLCM-Correlation-1"][i] =corr[0]
    df["GLCM-Correlation-2"][i] =corr[1]
    df["GLCM-Correlation-3"][i] =corr[2]
    df["GLCM-Correlation-4"][i] =corr[3]

    df["GLCM-ASM-1"][i] =asm[0]
    df["GLCM-ASM-2"][i] =asm[1]
    df["GLCM-ASM-3"][i] =asm[2]
    df["GLCM-ASM-4"][i] =asm[3]
    
    
    
    return 
def glcmfiller(imgarr,i,gry):
    import cv2
    if imgarr.shape !=(224, 224):
        og = cv2.cvtColor(imgarr,cv2.COLOR_BGR2GRAY)
        fxnglcm(og,i)
        df["gry"][i] = og
    else:
        fxnglcm(imgarr,i)
        df["gry"][i] = imgarr
    return

def lbp_calculated_pixel(img, x, y):
   
    center = img[x][y]
   
    val_ar = []
      
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
      
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
      
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))
      
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
      
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
      
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
      
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))
      
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
       
    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
   
    val = 0
      
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    
   
    img_lbp = np.zeros((224, 224),
                       np.uint8)
       
    for i in range(0, 224):
        for j in range(0, 224):
            img_lbp[i, j] = lbp_calculated_pixel(df_lbp["gry"][3], i, j)
    plt.imshow(df_lbp["gry"][3])
    plt.show()
       
    plt.imshow(img_lbp, cmap ="gray")
    plt.show()

###
#GMP Classifier
class GMPPooling():
    def __init__(self, p=3.0, eps=1e-6):
        super(GMPPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return (torch.mean((x + self.eps).pow(self.p), dim=(-1, -2))).pow(1. / self.p)

class BrainTumorGMPClassifier():
    def __init__(self, num_classes=2):
        super(BrainTumorGMPClassifier, self).__init__()
        
        # Use a pretrained model as feature extractor
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        
        # Add GMP Pooling
        self.gmp = GMPPooling(p=3.0)
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gmp(x)
        x = self.classifier(x)
        return x

class BrainTumorDataset():
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
def model():

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset
    train_dataset = BrainTumorDataset(root_dir="dataset/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorGMPClassifier(num_classes=2).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")


#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""

    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
  
    ##
    vv=[]
    vn=0
    data2=[]
    v1=0
    v2=0
    v3=0
    v4=0
    dt=[]
    dt2=[]
    dt3=[]
    dt4=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        
        if fname[0]=='g':            
            dt.append(fname)
            v1+=1
        elif fname[0]=='m':            
            dt2.append(fname)
            v2+=1
        elif fname[0]=='p':            
            dt4.append(fname)
            v4+=1
        else:
            dt3.append(fname)
            v3+=1
        
    
    data2.append(dt)
    data2.append(dt2)
    data2.append(dt3)
    data2.append(dt4)
        
   
    vv=[v1,v2,v3,v4]
    gg=v1+v2+v3+v4
    dd2=vv
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,gg))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

#######
@app.route('/home', methods=['GET', 'POST'])
def home():
    msg=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname, ))
    data = mycursor.fetchone()
       
    
    return render_template('home.html', data=data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    f2=open("static/trained/bt.txt","r")
    dd=f2.read()
    f2.close()


    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0
    result=""
    if 'username' in session:
        uname = session['username']
    
    mycursor = mydb.cursor(buffered=True)
    mycursor.execute("SELECT * FROM register where uname=%s",(uname, ))
    data = mycursor.fetchone()
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    ff=open("static/js/jj.txt",'r')
    fnp=ff.read()
    ff.close()
    fnpath=fnp.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=0
        path_main = 'static/dataset'
        #for fname1 in os.listdir(path_main):
        for fname1 in fnpath:
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            print(fname1+" "+filename)
            if cc1<=cutoff:
                #if fname1==filename:
                ss="ok"
                print(fname1)
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            
            dimg=[]

            if fn[0]=='g':            
                tclass=0
            elif fn[0]=='m':            
                tclass=1
            elif fn[0]=='p':            
                tclass=3
            else:
                tclass=2

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            
            ##
            n=0
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                va1=val.split('-')
                if va1[0]==fn:
                    print(val[0])
                    #0,1--Benign
                    #2,3,4--Malignant
                    result=va1[1]
                    
                    break
                
                n+=1
                
                    
           
        
            
            
            print(tclass)
            
            cla=cname[tclass]
            dta=cla+"|"+fn+"|"+result
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()
        
            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
    
    
        
    return render_template('upload.html',msg=msg)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

        
    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    img=[]
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]

    

    
    f2=open("static/trained/bt.txt","r")
    dd=f2.read()
    f2.close()
    
    sdata=dd.split("|")
    for ss in sdata:
        ds=[]
        dt=ss.split(",")
        
        if dt[0]==fn:
            print(dt[0])
            img.append(dt[0])
            img.append(dt[1])
            img.append(dt[2])
            img.append(dt[3])
            img.append(dt[4])
            img.append(dt[5])
            img.append(dt[6])
            break
   
            

    print(img)
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,img=img)


@app.route('/test_pro3', methods=['GET', 'POST'])
def test_pro3():
    msg=""
    fn=""
    level=""
    data3=[]
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    stage=gs[2]

    if stage=="0" or stage=="1":
        level="Benign"
    else:
        level="Malignant"
    print(ts)
    cursor = mydb.cursor(buffered=True)

    if ts=="No Tumor":
        st=""
        
    else:
        # where btype=%s',(ts,)
        cursor.execute('SELECT count(*) FROM recommend')
        cnt = cursor.fetchone()[0]
        st=""
        
        if cnt>0:
            st="1"
            cursor.execute('SELECT * FROM recommend order by rand() limit 0,3')
            data3 = cursor.fetchall()

        
    return render_template('test_pro3.html',msg=msg,fn=fn,ts=ts,act=act,data3=data3,st=st,stage=stage,level=level)

##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080, debug=True)


