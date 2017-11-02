import numpy as np
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import sklearn.cross_validation

np.random.seed(20170508)

X_test = np.load('saekano_face_data.npy')
T_target = np.load('saekano_face_label.npy')

x_train,x_test,t_train,t_test = sklearn.cross_validation.train_test_split(X_test,T_target)

model = Sequential()

model.add(Convolution2D(96,3,3,border_mode='same',input_shape=(3,32,32)))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(5))
model.add(Activation('softmax'))

init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate,decay=0.0,momentum=0.9,nesterov=False)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['acc'])
early_stopping = EarlyStopping(monitor='var_loss',patience=3,verbose=0,mode='auto')
lrs = LearningRateScheduler(0.01)

hist = model.fit(x_train,t_train,batch_size = 128,nb_epoch=50,validation_split=0.1,verbose=1)

model_json_str = model.to_json()
open('saekano_face_model.json','w').write(model_json_str)
model.save_weights('saekano_face_model.h5')
