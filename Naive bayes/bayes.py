import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import glob
import cv2
import pathlib 

# Loading the data
path = 'paddy-disease-classification/train_images/'
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

x_train=x_train.reshape(len(x_train),200*200*3)
x_test=x_test.reshape(len(x_test),200*200*3)

nb=GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb=nb.predict(x_test)

print(classification_report(y_pred_nb,y_test))
# Create a random forest model
nb = GaussianNB()

# Fit the model to the training data
nb.fit(x_train, y_train)

# Predict the labels for the test data
y_pred_nb = nb.predict(x_test)

# Print the classification report
print(classification_report(y_pred_nb, y_test))

report = pd.DataFrame(classification_report(y_pred_nb, y_test, output_dict=True))
report.to_csv('classification_report.csv')

import pickle
# Save the trained model to a file
with open('bayes_model.pkl', 'wb') as file:
    pickle.dump(nb, file)
