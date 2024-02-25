import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import glob
import cv2
import pathlib 

# Loading the data
path = 'paddy-disease-classification/train_images/'
dataset=pathlib.Path(path)

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

# Load images and their labels
x, y = [], []
for disease_name, disease_images in image_dict.items():
    for image in disease_images:
        img = cv2.imread(str(image))
        re_img = cv2.resize(img, (200, 200))
        x.append(re_img)
        y.append(label_dict[disease_name])

x = np.array(x)
y = np.array(y)

# Splitting the data into training, testing, and validation sets
x_train, x_test_d, y_train, y_test_d = train_test_split(x, y, train_size=0.7, stratify=y)
x_test, x_val, y_test, y_val = train_test_split(x_test_d, y_test_d, test_size=0.4)

x_train = x_train.reshape(len(x_train), 200*200*3)
x_test = x_test.reshape(len(x_test), 200*200*3)

# Define hyperparameters grid
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Create KNN classifier
knn = KNeighborsClassifier()

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best model
best_knn = grid_search.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Predict the labels for the test data
y_pred_knn = best_knn.predict(x_test)

# Print the classification report
print(classification_report(y_pred_knn, y_test))

report = pd.DataFrame(classification_report(y_pred_knn, y_test, output_dict=True))
report.to_csv('classification_report.csv')

# Save the trained model to a file
import pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(best_knn, file)
