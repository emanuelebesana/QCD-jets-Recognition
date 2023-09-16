import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import unravel_index
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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

images = images.reshape((len(images),-1))

train_size = 0.8
test_size = 1-train_size
X_train,X_test,Y_train,Y_test = train_test_split(images,signal,train_size=train_size,test_size=test_size)


C_vals = np.logspace(-5,4,10)


"""#rbf kernel

gamma_vals = np.logspace(-5,4,10)

accuracy_scores = np.zeros((len(gamma_vals),len(C_vals)))
for i,GAMMA in enumerate(gamma_vals):
    for j,C in enumerate(C_vals):
        classifier = SVC(gamma=GAMMA,C=C)
        classifier.fit(X_train,Y_train)

        y_pred = classifier.predict(X_test)

        accuracy_scores[i][j]=accuracy_score(Y_test,y_pred)
        print("Gamma = ", GAMMA,"    C = ", C, "      done")



maximum_params = unravel_index(accuracy_scores.argmax(),accuracy_scores.shape)
print("Best case scenario is -> C: ", C_vals[maximum_params[1]],"  Degree: ", gamma_vals[maximum_params[0]],"  with test accuracy: ", np.max(accuracy_scores))

sns.set()

fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(accuracy_scores,annot=True,ax=ax,cmap="viridis")
ax.set_title("Test Accuracy",fontsize=15)
ax.set_xlabel("$C$",fontsize=15)
ax.set_ylabel("$\gamma$",fontsize=15)
ax.set_xticks(np.arange(0,len(gamma_vals))+0.5,labels = gamma_vals)
ax.set_yticks(np.arange(0,len(C_vals))+0.5,labels = C_vals)
save_fig("Test Accuracy SVM")
plt.show()
"""


"""#polynomial kernel

poly_vals = np.arange(3,7)

for i,poly_deg in enumerate(poly_vals):
    for j,C in enumerate(C_vals):
        classifier = SVC(kernel='poly',degree=poly_deg,C=C) #polynomial kernel
        classifier.fit(X_train,Y_train)

        y_pred = classifier.predict(X_test)

        accuracy_scores[i][j]=accuracy_score(Y_test,y_pred)
        print("Degree = ", poly_deg,"    C = ", C, "      done")



maximum_params = unravel_index(accuracy_scores.argmax(),accuracy_scores.shape)
print("Best case scenario is -> C: ", C_vals[maximum_params[1]],"  Degree: ", poly_vals[maximum_params[0]],"  with test accuracy: ", np.max(accuracy_scores))
"""


"""#linear kernel

linear_scores = np.zeros(len(C_vals))
for j,C in enumerate(C_vals):
    Linear_classifier = SVC(kernel='linear',C=C)
    Linear_classifier.fit(X_train,Y_train)

    y_pred = Linear_classifier.predict(X_test)

    linear_scores[j] = accuracy_score(Y_test,y_pred)


print("Accuracy score for linear kernel -> ", np.max(linear_scores))
print("\nFor C val -> ", C_vals[np.argmax(linear_scores)])
"""
