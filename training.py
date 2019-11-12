####Importing the libraries####
import os
import pickle
import tensorflow as tf
from keras import backend as K
from preprocessing import get_data
from classification_network import compile_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Model
from numpy.random import seed
from tensorflow import set_random_seed

####Setting seeds####
def set_seeds():
	seed(18)
	set_random_seed(25)
	
####Training the model####
def train_model(model, dataset, n_epoch, n_batch, save_file, data_path):
	####Set training parameters####
	nb_epochs = n_epoch
	nb_batch = n_batch
	if not os.path.exists("weights"):
		os.makedirs("weights")
	save_path=os.path.join("weights",save_file)+".h5"
	
	checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
	earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
	tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
	
	####Get data####
	[X_train, y_train, X_test, y_test] = dataset
	
	####Training model####
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = nb_epochs, batch_size=nb_batch, callbacks = [earlyStopping, lr_reduce, checkpoint, tbCallBack], verbose=2)

	####Saving history####
	if not os.path.exists("history"):
		os.makedirs("history")
	with open(os.path.join('history',save_file), 'wb') as file_pi:
		pickle.dump(history.history, file_pi)