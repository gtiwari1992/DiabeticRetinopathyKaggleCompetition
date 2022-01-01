import os
import sys
import csv
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn . preprocessing import StandardScaler , LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

from keras.preprocessing import image


input_dir = os.getcwd()
plot_dir = os.getcwd()

trainlabels = pd.read_csv('trainLabels.csv')


path=r'C:\Users\gtiwa\.spyder-py3\image3'

classes = ["No DR","Mild","Moderate","Severe","Proliferative DR"]


training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img),cv2.COLOR_BGR2RGB)
    # pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    try:
        pic = cv2.resize(pic,(200,200))
    except:
        break    
    pic = np.array(pic)
    # pic = pic.astype('float32')
    training_data.append(pic)


np.save(os.path.join(path,'features'),(training_data))
np.save(os.path.join(path,'features'),np.array(training_data))

saved = np.array(training_data)

print(len(training_data))



df1 = pd.read_csv('../input/train5/trainLabels1.csv')

df2 = df1[['image', 'level']]

df3 = df2[['level']]




X = saved





classes = ["No DR","Mild","Moderate","Severe","Proliferative DR"]

def plot_sample(X,Y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[Y.loc[index][0]])
    

data = df3
class_labels = [0,1,2,3,4]
data = data[data['level'].isin(class_labels)]
Y = (data['level'].values)

X_train ,X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.5 ,random_state =3, shuffle = False)



ann = models.Sequential([
    layers.Flatten(input_shape=(200,200,3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(5, activation='softmax')
    ])

ann.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

ann.fit(X_train,Y_train, epochs=10)

y_pred = ann.predict(X_test)

print(y_pred)

y_pred_classes = [np.argmax(element) for element in y_pred]

print(y_pred_classes)

acc = np.mean(y_pred_classes == Y_test)

k = ann.evaluate(X_test,Y_test)

print("Below are the results for ANN")

print(k)

print (confusion_matrix(y_pred_classes,Y_test))

print("Classification Report: \n", classification_report(Y_test,y_pred_classes))

conf1 = confusion_matrix(y_pred_classes,Y_test, class_labels)

disp1 = ConfusionMatrixDisplay(confusion_matrix = conf1,display_labels=classes)

disp1.plot()

print("Classification Report: \n", classification_report(Y_test,y_pred_classes))


cnn = models.Sequential([
    layers.Conv2D(filters= 32,kernel_size=(3,3), activation= 'relu', input_shape = (200,200,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters= 32,kernel_size=(3,3), activation= 'relu', input_shape = (200,200,3)),
    layers.MaxPooling2D((2,2)),    
       
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
    ])

accuracy = []
epochs= []

ypredictions = []

def CNN(X_train, Y_train,X_test,Y_test ):
    
    for i in range (5,30):

        cnn.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

        cnn.fit(X_train,Y_train, epochs = i)

        y_pred1 = cnn.predict(X_test)

        print(y_pred1)

        y_pred_classes1 = [np.argmax(element) for element in y_pred1]

        print(y_pred_classes1)
        
        acc1 = np.mean(y_pred_classes1 == Y_test)
                
        epochs.append(i)
        
        ypredictions.append(y_pred_classes1)
        
        accuracy.append(acc1)
    
    print(accuracy)
    print(epochs)
    
    print("The maximum accuracy obtained is",max(accuracy))
    
    plt.plot(epochs,accuracy)
    plt.title('Epochs to Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
    return(epochs,accuracy,ypredictions[accuracy.index(max(accuracy))])



(epochs, accuracy,y_pred_classes1)= CNN(X_train, Y_train,X_test,Y_test )

conf = confusion_matrix(y_pred_classes1,Y_test, class_labels)

disp = ConfusionMatrixDisplay(confusion_matrix = conf,display_labels=classes)

disp.plot()

plt.show()
