import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import unravel_index
import os
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.utils import to_categorical


PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results_FigureResults"
DATA_ID = "DataFiles"

def image_path(fig_id):
    return os.path.join(FIGURE_ID,fig_id)
def data_path(dat_id):
    return os.path.join(DATA_ID,dat_id)
def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')



'''
The first part of the program loads the data into the program. The data was taken from the website https://zenodo.org/record/269622#.Yf0ont_MLIX, and 6000 out of the
872666 images were converted to the images.csv file. The corresponding 6000 signals/targets are found in the signal.csv file.
Since the images are a (6000,25,25) file, the way the data is loaded is done by looking at how the images are arranged in the images.csv file.
'''

np.random.seed(3)

imgs = pd.read_csv('images.csv',header=None)
imgs = imgs.to_numpy()

sgnl = pd.read_csv('signal.csv',header=None)
sgnl = sgnl.to_numpy() #(6000,1) column array
signal = sgnl[:,0]   #(6000,) array

n_pixels = 25
n_images = 6000

#here we put the pictures in a (n_images,n_pixels,n_pixels) 3D array
i=0
b=0
images = np.zeros((n_images,n_pixels,n_pixels))
while i<len(imgs[0]):
    newimage = np.zeros((n_pixels,n_pixels))
    for c in range(n_pixels):
        newimage[:,c] = imgs[:,c+i]

    images[b] = newimage

    b+=1
    i+=n_pixels


#actual neural network

images = images/np.max(images)  #normalize pixel values between 0 and 1. images=(6000,25,25)
images = images[:,:,:,np.newaxis]
labels = to_categorical(signal)  #labels = (6000,2)

print("-------done uploading images and converting data strucutres --------- returning to sleep ")

train_size = 0.8
test_size = 1-train_size
X_train,X_test,Y_train,Y_test = train_test_split(images,labels,train_size=train_size,test_size=test_size)


def create_convolutional_nn_keras(input_shape,receptive_field,n_filters1,n_filters2,n_filters3,n_neurons_connected,n_categories,eta,lmbd):

    model = Sequential()

    model.add(layers.Conv2D(n_filters1,(receptive_field,receptive_field),input_shape=input_shape,padding='same',activation='relu',kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(n_filters2,(receptive_field,receptive_field),padding='same',activation='relu',kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(n_filters3,(receptive_field,receptive_field),padding='same',activation='relu',kernel_regularizer=regularizers.l2(lmbd)))

    model.add(layers.Flatten())
    model.add(layers.Dense(n_neurons_connected,activation='relu',kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.Dense(n_categories,activation='softmax',kernel_regularizer=regularizers.l2(lmbd)))

    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model


epochs= 5
batch_size = 20
input_shape=X_train.shape[1:4] #25x25x1 is the (height,width,channels),channels=1 because no RGB, just grayscale
receptive_field=3
n_filters1 = np.arange(30,41)
n_filters2 = np.arange(30,41)
n_filters3 = 35
n_neurons_connected= 70
n_categories=2
eta_vals=np.logspace(-4,1,7)
lmbd_vals=np.logspace(-4,1,7)

print("-------done defining characteristics of the network --------- returning to sleep")


test_accuracy_parameters = np.zeros((len(eta_vals), len(lmbd_vals)))
for i,eta in enumerate(eta_vals):
    for j,lmbd in enumerate(lmbd_vals):
        CNN = create_convolutional_nn_keras(input_shape,receptive_field,35,35,n_filters3,n_neurons_connected,n_categories,eta,lmbd)
        CNN.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,verbose=1)

        test_accuracy_parameters[i][j] = CNN.evaluate(X_test, Y_test)[1]


maximum_params = unravel_index(test_accuracy_parameters.argmax(),test_accuracy_parameters.shape)
min_eta = eta_vals[maximum_params[0]]
min_lmbd = lmbd_vals[maximum_params[1]]

print("-------done grid searching the parameters --------- returning to sleep")



#min_eta,min_lmbd are the eta and lmbd values that result in the minimum test error

#now do filters grid search

test_accuracy_filters = np.zeros((len(n_filters1), len(n_filters2)))
for i,n_filt1 in enumerate(n_filters1):
    for j,n_filt2 in enumerate(n_filters2):
        CNN = create_convolutional_nn_keras(input_shape,receptive_field,n_filt1,n_filt2,n_filters3,n_neurons_connected,n_categories,min_eta,min_lmbd)
        CNN.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,verbose=1)

        test_accuracy_filters[i][j] = CNN.evaluate(X_test, Y_test)[1]


maximum_filters = unravel_index(test_accuracy_filters.argmax(),test_accuracy_filters.shape)
min_n_filters1 = n_filters1[maximum_filters[0]]
min_n_filters2 = n_filters2[maximum_filters[1]]


print("-------done grid searching the filters--------- returning to sleep")

sns.set()

fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(test_accuracy_filters,annot=True,ax=ax,cmap="viridis")
ax.set_title("Test Accuracy",fontsize=15)
ax.set_xlabel("1st layer filters",fontsize=15)
ax.set_ylabel("2nd layer filters",fontsize=15)
ax.set_xticks(np.arange(0,len(n_filters1))+0.5,labels = n_filters1)
ax.set_yticks(np.arange(0,len(n_filters2))+0.5,labels = n_filters2)
save_fig("Test Accuracy filters CNN3Pooling")



print("\nFinal parameters: eta -> ", min_eta)
print("\nFinal parameters: lambda -> ", min_lmbd)
print("\nFinal parameters: n_filters1 -> ", min_n_filters1)
print("\nFinal parameters: n_filters2 -> ", min_n_filters2)

print("\nFinal test accuracy is -> " ,np.max(test_accuracy_filters))
