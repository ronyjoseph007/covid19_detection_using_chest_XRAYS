from django.shortcuts import render,HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from tensorflow.python.keras.backend import set_session

import requests
import h5py
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
import tensorflow as tf


sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

set_session(sess)
with graph.as_default():
 covimod = load_model('./models/coviup.h5')





# global
label_dict=['Covid19 Negative','Covid19 Positive']
img_width,img_height=100,100


# Create your views here.
def covipred(req):

    return render(req,'index.html')

def covidtest(req):

    return render(req,'covid_test.html')

def case(req):

   data = requests.get('https://covid19.mathdro.id/api/countries/INDIA').json()
   confirmed=data['confirmed']['value']
   recovered=data['recovered']['value']
   deaths=data['deaths']['value']

   cases={
       "confirmed":confirmed,
       "recovered": recovered,
       "deaths":deaths
   }


   return render(req,'data.html',context=cases)

def predict(req):

    if req.method =='POST' and req.FILES['myfile']:
        myfile=req.FILES['myfile']
        fs=FileSystemStorage()
        filename=fs.save(myfile.name,myfile)
        fileurl = fs.url(filename)
        image= '.'+fileurl
        img=load_img(image,target_size=(img_height,img_width))

        img=np.array(img)
        if (img.ndim == 3):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray=img


        gray=gray/255
        img_re = gray.reshape(1,img_width, img_height,1)


        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            result = covimod.predict([img_re])


        print(result)

        if (result[0][1]<0.9):
            print('neg')
            res=0
            pred=label_dict[0]
        else:
            print('pos')
            res=1
            pred=label_dict[1]


    context={"result":pred,"result_num":res}
    return render(req,'covid_test_result.html',context)
