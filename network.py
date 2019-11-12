####Importing the libraries####
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D
from keras.layers import Input, Add, Concatenate, ELU, ReLU
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD

####Desiging the residual block####
def residual_block(mod, k_, f_in, f_out, strides = (1,1), useshortcut = False):
    
    shortcut = mod

    mod = SeparableConv2D(f_in, kernel_size = k_, strides=(1,1), padding = "same")(mod)
    mod = BatchNormalization()(mod)
    mod = ELU()(mod)

    mod = SeparableConv2D(f_in, kernel_size = k_, strides=strides, padding = "same")(mod)
    mod = BatchNormalization()(mod)
    mod = ELU()(mod)

    mod = SeparableConv2D(f_out, kernel_size = k_, strides=(1,1), padding = "same")(mod)
    mod = BatchNormalization()(mod)

    if strides != (1,1) or useshortcut:
        shortcut = SeparableConv2D(f_out, kernel_size = k_, strides=strides, padding = "same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    mod = Add()([shortcut, mod])
    mod = ReLU()(mod)

    return mod

####Designing the classification network####
def model_build(in_, k_):
    
    mod = SeparableConv2D(16, kernel_size = k_, strides=(1,1), padding="same")(in_)
	
    mod = BatchNormalization()(mod)
	
    mod = ReLU()(mod)
    
    mod = residual_block(mod, k_, 16, 32, useshortcut=True)
    
    mod = MaxPooling2D()(mod)
    
    mod = residual_block(mod, k_, 32, 48, useshortcut=True)
    
    mod = MaxPooling2D()(mod)
    
    mod = residual_block(mod, k_, 48, 64, useshortcut=True)
    
    mod = MaxPooling2D()(mod)
    
    mod = residual_block(mod, k_, 64, 96, useshortcut=True)
    
    mod = GlobalAveragePooling2D()(mod)
	
    mod = Dense(256, activation="relu")(mod)
	
    mod = Dropout(0.5)(mod)
	
    mod = Dense(2, activation="softmax")(mod)
    
    return mod
	
def compile_model():
	in_=Input((224, 224, 3))
	k_ = (4,4)
	pred = model_build(in_, k_)
	model = Model(input = in_, output = pred)
	model.compile(optimizer = RMSprop(), metrics = ["accuracy"], loss="binary_crossentropy")
	return model