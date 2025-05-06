################################################## Import Libraries ####################################################
# Import tensorflow
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # For multi-class Classification (one-hot encoded labels)
from tensorflow.keras.metrics import Precision  # Import Precision to keep track

# Import Sklearn
from sklearn.model_selection import train_test_split  # to split data into test and train samples
from sklearn.preprocessing import StandardScaler  # normalizing (standardizing) the dataset.
from sklearn.metrics import (classification_report,
                             confusion_matrix,  # For Confusion Matrix
                             roc_curve,  # For Receiver Operating Characteristic curve
                             auc,  # Area Under the Curve
                             roc_auc_score  # For Keeping ROC and AUC Scores
                             )

# Import os
import os

# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np

# Custom Functions
from Feature_Generation.feature_generator import Data_Store
from ML_Generation.Neural_Network_Class import NN_Classifier

# Clean Data ###########################################################################################################
data = pd.read_csv(r"C:\Users\I'm the best\Documents\a\Infrastructure\Feature_Generation\BTCUSDT3600.csv")

#Calculate Features ####################################################################################################
SMAs = Data_Store(data, 'SMA', {"length": [40, 60, 80], "source": 'close'} )

# Combine everything into Input data ###################################################################################
# A expert level merging method
Input_Data = data.copy()
Input_Data[SMAs.columns] = SMAs

# Those are temporarily fixes to make the script run on my pc because for me close and date are in lower case not upper case
Input_Data = Input_Data.rename(columns = {"close": 'Close', 'date': "Date"})
Input_Data["Date"] = pd.to_datetime(Input_Data["Date"])
Input_Data.to_csv("Testing_Data", index = False)

## Feed Models #########################################################################################################
NN_Classifier(
    #Dataset
    Data = Input_Data,
    t_n = 25,
    ####### Creating the Model #########
    layer_sizes=np.arange(10, 30, 5),
    activation='relu',
    output_activation='sigmoid',
    output_units = 1,
    dropout_mode='uniform',
    dropout=0.2,
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision')],
    ####### Training the Model ##########
    epochs=30,
    batch_size=64
    )


#print(features)

