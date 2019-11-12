####Importing necessary libraries####
import os
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix

####plotting roc curve####
def plot_roc_curve(model, X, y, weight_path, save_file, label = None):
    ####Loading weights of units in model####
    model.load_weights(weight_path)
    
    ####Getting prediction scores####
    y_pred_probs = model.predict(X)
    y_labels = np.argmax(y, axis = 1)
    fpr, tpr, thresholds = roc_curve(y_labels, y_pred_probs[:, 1])
    fig = plt.figure(figsize=(15,15))
    plt.plot(fpr, tpr, linewidth=2, label = "AUC score = "+str(roc_auc_score(y_labels, y_pred_probs[:,1])))
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", title="AUC score")
    if not os.path.exists('visualization'):
        os.makedirs('visualization')
    fig.savefig(os.path.join("visualization",save_file+".png"), transparent=True, dpi=5*fig.dpi)
    plt.show()
    
####Plotting confusion matrix####
def plot_conf_mat(model, X, y, weight_path, save_file):
    y_true_labels = np.argmax(y, axis = 1)
    y_pred_labels = np.argmax(model.predict(X), axis = 1)

    confmat = confusion_matrix(y_true_labels, y_pred_labels)
    fig, _ = plot_confusion_matrix(conf_mat= confmat, figsize=(15,15))
    if not os.path.exists('visualization'):
        os.makedirs('visualization')
    fig.savefig(os.path.join("visualization",save_file+".png"), transparent=True, dpi=5*fig.dpi)
    plt.show()