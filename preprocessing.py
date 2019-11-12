####Importing necessary libraries####
import numpy as np
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit as s_split
from keras.utils.np_utils import to_categorical

####Defining the preprocessing method that saves the processed image data to a remote folder####
def set_data(path,savepath):
    for nextDir in os.listdir(path):
        if not nextDir.startswith('.'):
            temp = path + "/" + nextDir
            clahe = cv2.createCLAHE(clipLimit = 1.5, tileGridSize = (8,8))
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + "/" + file,0)
                if img is not None:
                    img = cv2.bilateralFilter(img, 8, 50, 50)
                    img = clahe.apply(img)
                    img=cv2.resize(img,(224,224))
                    p=os.path.join(savepath,nextDir)
                    if not os.path.exists(p):
                        os.makedirs(p)
                    cv2.imwrite(os.path.join(p,file),img)
	
####Defining a method to read the preprocessed data from the remote folder####	
def get_data(path):
    X=[]
    y=[]
    for nextDir in os.listdir(path):
        if not nextDir.startswith('.'):
            label=2
            if nextDir in ['NORMAL']:
                label=0
            elif nextDir in ['PNEUMONIA']:
                label=1
            temp=path+"/"+ nextDir
            for file in tqdm(os.listdir(temp)):
                img=cv2.imread(temp +"/" + file)
                if img is not None:
                    X.append(img)
                    y.append(label)
    return X,y    

####Shuffling the data using stratified shuffle spilt####
def shuffle_data(X,y):
	X=np.asarray(X)
	y=np.asarray(y)
	split= s_split(n_splits=1,test_size=0.2, random_state=18)
	X_train=[]
	X_test=[]
	y_train=[]
	y_test=[]
	for train_id,test_id in split.split(X,y):
		X_train.append(X[train_id])
		X_test.append(X[test_id])
		y_train.append(y[train_id])
		y_test.append(y[test_id])
	X_train=np.asarray(X_train,dtype="float32")[0]
	y_train=np.asarray(y_train)[0]
	X_test=np.asarray(X_test,dtype="float32")[0]
	y_test=np.asarray(y_test)[0]
	X_train /= 255
	X_test /= 255
	y_train = to_categorical(y_train, 2)
	y_test = to_categorical(y_test, 2)
	return [X_train, y_train, X_test, y_test]