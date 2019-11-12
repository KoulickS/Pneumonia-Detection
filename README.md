# A deep learning approach to automate diagnosis of Pneumonia from CXR images
A deep residual network implementing separable convolution to diagnose Pneumonia from CXR images. Here, we discuss about the script used for data preprocessing and model development.

# Data used for development
The data useed can be found at Mendeley data. Link: https://data.mendeley.com/datasets/rscbjbr9sj/2. This repository does not include the RAW data.

# Data preprocessing
The image data was processed for noise removal and CLAHE enhancement. These preprocessing steps can be applied using the set_data() method defined within the "preprocessing.py" script. This method saves the processesd data to a remote folder. This data can be read using the get_data() method. This method requires the normal CXR images to be saved with a folder within a folder named "NORMAL" and the Pneumonia positive images to be placed with a folder named "PNEUMONIA".

# Model development
The model was developed using residual learning paradigms. Additionally, separable convolution was used in order to optimize paramter cost and time complexity. The compiled model used for the diagnosis purpose can be obtained using the compile_model() method defined within the "network.py" script. 

# Model training
The model's training parameters have ben defined in the "train_model()" method. Model can be training using the same method. One must however, refer to the previously mentioned scripts along with this script in order to gain insight into the parameters for the functions.

# Performance of the network
The model's performance can be visualized using the "performance.py" script which includes the various methods necessary in order to illustrate the ROC curve and the confusion matrix. One may also choose to visualize the grad-CAMs for the convolution layers using keras-vis library.

# Requirements
Keras(v2.2.2) 
keras-vis(v0.4.1)(optional)
scikit-learn(v0.19.2)  
