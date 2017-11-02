# -*- coding: utf-8 -*-
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
import sklearn.cross_validation
import cv2 as cv
from tweepy import *
import urllib.request
from PIL import Image
import os
import skimage

np.random.seed(20170508)

#PreparingTwitterAPI
CONSUMER_KEY = '#####'
CONSUMER_SECRET = "#####"
auth = OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
ACCESS_TOKEN = '#####'
ACCESS_SECRET = '#####'
auth.set_access_token(ACCESS_TOKEN,ACCESS_SECRET)
api = API(auth)

#learningResult
X_test = np.load('saekano_face_data.npy')
Y_target = np.load('saekano_face_label.npy')
model = model_from_json(open('saekano_face_model.json').read())
model.load_weights('saekano_face_model.h5')
init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate,decay=0.0,momentum=0.9,nesterov=False)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['acc'])

cascade_path = './lbpcascade_animeface.xml'
color = (255,255,255)

#画像を判別
def judgement():
    image = cv.imread('given.jpg')
    if not image is None:
        image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image_gray = cv.equalizeHist(image_gray)
        cascade = cv.CascadeClassifier(cascade_path)
        facerect = cascade.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=3, minSize=(50, 50))
        if len(facerect) > 0:
            for rect in facerect:
                croped = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                cv.imwrite('face.jpg',croped)

            image = cv.imread('face.jpg')
            image = cv.resize(image,(32,32))
            image = image.transpose(2,0,1)
            image = image/255
            image = image.reshape(1,3,32,32)

            for i in range(4):
                sample_target = np.array([i])
                score = model.evaluate(image,sample_target,verbose=0)
                if score[1] == 1.0:
                    break
        else:
            i = 5
    else:
        i = 6
    return i

class StreamListener(StreamListener):
    def on_status(self,status):
        if status.in_reply_to_screen_name == '#####':
            medias = status.entities['media']
            m = medias[0]
            media_url = m['media_url']
            try:
                urllib.request.urlretrieve(media_url,'given.jpg')
            except IOError:
                print('Error')
                
            figure = judgement()
            if figure == 0:
                reply = '@'+status.author.screen_name+u' これは英梨々だね'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 1:
                reply = '@'+status.author.screen_name+u' これは私だね、いつの間に撮ったの？'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 2:
                reply = '@'+status.author.screen_name+u' 見たことあるような、ないような....'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 3:
                reply = '@'+status.author.screen_name+u' うわ、安芸くんだ'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 4:
                reply = '@'+status.author.screen_name+u' これは霞ヶ丘先輩だね'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 5:
                reply = '@'+status.author.screen_name+u' 顔が分からないよ〜'
                api.update_status(status=reply,in_reply_to_status_id=status.id)
            elif figure == 6:
                reply = '@'+status.author.screen_name+u' ごめんね、読み込めないみたい'
                api.update_status(status=reply,in_reply_to_status_id=status.id)

            if os.path.isfile('given.jpg'):
                os.remove('given.jpg')

            if os.path.isfile('face.jpg'):
                os.remove('face.jpg')

stream = Stream(auth,StreamListener(),secure=True)
print('Streaming!')
stream.userstream()
