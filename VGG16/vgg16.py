import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import glob
import cv2
import pathlib 

# Loading the data
path = '../paddy-disease-classification/train_images/'
dataset=pathlib.Path(path)
images=list(dataset.glob('*/*.jpg'))

blast=list(dataset.glob('blast/*'))
dead_heart=list(dataset.glob('dead_heart/*'))
hispa=list(dataset.glob('hispa/*'))
normal=list(dataset.glob('normal/*'))
tungro=list(dataset.glob('tungro/*'))


# Adding labels to the images
label_dict={
    'blast':0,
    'dead_heart':1,
    'hispa':2,
    'normal':3,
    'tungro':4
}
image_dict={
    'blast':list(dataset.glob('blast/*')),
    'dead_heart':list(dataset.glob('dead_heart/*')),
    'hispa':list(dataset.glob('hispa/*')),
    'normal':list(dataset.glob('normal/*')),
    'tungro':list(dataset.glob('tungro/*'))
}

x,y=[],[]

for dis_name,dis_image in image_dict.items():
    for image in dis_image:
        img = cv2.imread(str(image))
        re_img= cv2.resize(img,(200,200))
        x.append(re_img)
        y.append(label_dict[dis_name])
        
x=np.array(x)
y=np.array(y)

# Splitting the data into training, testing and validation set
x_train,x_test_d,y_train,y_test_d=train_test_split(x,y,train_size=0.7,stratify=y)
x_test,x_val,y_test,y_val=train_test_split(x_test_d,y_test_d,test_size=0.4)


# Creating the model using VGG16
num_classes=5
model = Sequential()
model.add(VGG16(weights='imagenet', include_top=False,pooling='avg',input_shape=(200,200,3)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable=False             
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical

num_epochs = 20
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
history=model.fit(x_train,y_train_cat,validation_data=(x_val,y_val_cat),epochs=num_epochs)

model_hist=pd.DataFrame(history.history)

model_hist.to_csv('model_hist.csv',index=False)

model.save('paddy_disease_model.h5')

y_test_cat=to_categorical(y_test,num_classes=num_classes)
y_pred_vgg16=model.predict(x_test)
y_pred_vgg16=np.argmax(y_pred_vgg16,axis=1)

print(classification_report(y_pred_vgg16,y_test))