#import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from tensorflow.keras.models import Sequential,load_model
from flask import Flask,render_template,request,flash,redirect
from werkzeug.utils import secure_filename
import os

app=Flask(__name__)
#app.config['IMAGE_UPLOADS']='D:\\DL-projects\\weather_prediction\\static\\images'
app.config['IMAGE_UPLOADS']='D:\\DL-projects\\weather_prediction\\static\\images'
app.config['ALLOWED_EXTENSIONS']=['.jpg','.jpeg']

model = load_model('model.h5')
def newImage(path,x,y):
    imgs=[]
    img=Image.open(path)
    # img=img.convert('L')
    img=img.resize(size=(x,y))
    img=np.array(img,dtype=np.float16)
    # img=img.reshape(img.shape[0],img.shape[1],1)
    imgs.append(np.array(img))
    return np.array(imgs)


def predict(path):
    x_new=newImage(path,32,32)
    y_pred=model.predict(x_new)
    y_pred=np.argmax(y_pred)
    # print(y_pred)
    if y_pred==0:
        return "cloudy"
    elif y_pred==1:
        return "foggy"
    elif y_pred==2:
        return "rainy"
    elif y_pred==3:
        return "shine"
    elif y_pred==4:
        return "sunrise"
  

# print(predict('rain_3.jpg'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def detect():
    if request.method=="POST":
        image=request.files['file']
        print(image)
        extension=os.path.splitext(image.filename)[1]
        if extension not in app.config['ALLOWED_EXTENSIONS']:
            return "only .jpg,.jpeg extensions are allowed"
        if image:
            image.save(os.path.join(app.config['IMAGE_UPLOADS'],secure_filename(image.filename)))
            pred=predict(os.path.join(app.config['IMAGE_UPLOADS'],secure_filename(image.filename)))
            # data={
            #     'prediction':f'{pred}',
            #     'file':f"{secure_filename(image.filename)}"
            # }
        #     return render_template('detect.html',data=data)
        # else:
        #     return "File Not Selected"
            return pred

if __name__=='__main__':
    app.run(debug=True)