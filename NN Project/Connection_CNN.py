################################################## Import Libraries ####################################################
# Import tensorflow
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # For multi-class Classification (one-hot encoded labels)
from tensorflow.keras.metrics import Precision  # Import Precision to keep track
from tensorflow.keras import regularizers

# Import Sklearn
from sklearn.model_selection import train_test_split  # to split data into test and train samples
from sklearn.preprocessing import StandardScaler  # normalizing (standardizing) the dataset.
from sklearn.metrics import (classification_report,
                             confusion_matrix,  # For Confusion Matrix
                             roc_curve,  # For Receiver Operating Characteristic curve
                             auc,  # Area Under the Curve
                             roc_auc_score  # For Keeping ROC and AUC Scores
                             )

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Feature_Generation')))

# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np


import sys
import os

# Import the feature generator using an absolute import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Feature_Generation')))
from feature_generator import Data_Store
from Input_Data_Organizer import Input_Data_Organizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML_Generation', 'CNN')))
from Parallel_CNN_Class import Parallel_CNN_Classifier

# Clean Data ###########################################################################################################
#data = pd.read_csv("BTCUSDT3600.csv")

#Calculate Features ####################################################################################################
#SMAs = Data_Store(data, 'SMA', {"length": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], "source": 'Close'} )

Organizer = Input_Data_Organizer(

    # Data Used to Calculate Indicators
    Data='BTCUSDT3600_First_5k.csv', #BTCUSDT3600_First_1k , BTCUSDT3600_First_5k, BTCUSDT3600_Last_1k, MLGO_25_03_2025

    # Simple Moving Average--------------------
    Use_SMA=False,
    # Choose Length or Lengths for SMA
    SMA_Length=list(np.arange(3, 30, 1)),
    # Choose What to Column to use to Calculate SMA on
    SMA_Source='Close',

    # Simple Moving Average--------------------
    Use_EMA=False,
    # Choose Length or Lengths for SMA
    EMA_Length=[48],
    # Choose What to Column to use to Calculate SMA on
    EMA_Source=['Close'],

    # Returns--------------------
    Use_Return=False,
    # Choose Length or Lengths for Return
    Return_Length= 12,
    # Choose What to Column to use to Calculate Return on
    Return_Source='Close',

    # MACD
    Use_MACD=False,
    MACD_Signal = 9,
    MACD_Fast = 12,
    MACD_Slow = 26,
    MACD_Source='Close',

    # RSI
    Use_RSI = True,
    RSI_Length = [8],
    RSI_Source = 'Close',

    # Hawkes
    Use_Hawkes = True,
    Hawkes_Kappa = [0.1], # Decay factor
    Hawkes_Lookback = [168], # Lookback
    Hawkes_Source = ['High', 'Low', 'Close'], # Hawkes requires the order to be High, Low, Close always  ,

    # Reversability
    Use_Reversability = True,
    Reversability_Lookback = [48, 168],
    Reversability_Source = 'Close',

    # BBWP
    Use_BBWP = True,
    BBWP_Length = [13, 26],
    BBWP_Lookback = [168],
    BBWP_Source = ['Close'],
    BBWP_MA_Type = 'sma'
)

Input_Data = Organizer.Return_Data()

print(Input_Data)




## Feed Models #########################################################################################################
'''NN_Classifier(
    #Dataset
    Data = Input_Data,
    t_n = 6,
    ####### Creating the Model #########
    layer_sizes=[64, 32, 16],
    activation='relu',
    output_activation='sigmoid',
    output_units = 1,
    dropout_mode='uniform',
    dropout=0.3,
    learning_rate = 0.0001,
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision')],
    ####### Training the Model ##########
    epochs=100,
    batch_size=64
    )'''

Parallel_CNN_Classifier(
    # Dataset
    Data=Input_Data,
    t_n=25,
    include_ohlcv = False, # Whether to include OHLCV as features in the training of the model
    predict_return_positive = False,# Predicting Positive or Negative Return
    add_noise_to_training = True, # This decides whether to add noise to training on each epoch in an attempt to achieve more robust results
    plot_feature_correlation_matrix = False, # Whether to plot the feature correlation matrix

    ####### Creating the Model #########
    kernel_depth = 72,

    conv_branches=[
        [{"filters": 64, "kernel_size": 6}, {"filters": 64, "kernel_size": 3}],
        [{"filters": 64, "kernel_size": 24}, {"filters": 64, "kernel_size": 12}],
        [{"filters": 64, "kernel_size": 60}, {"filters": 64, "kernel_size": 36}]
    ],
    pool_type="max",  # "max" or "avg"
    pool_size=2,
    dropout_rate=0.3,
    dense_layer_dropout_rate = 0.3,

    dense_layers=[
        {"units": 256, "activation": "relu"},
        {"units": 64, "activation": "relu"},# 'kernel_regularizer': regularizers.l2(0.0001)}, # Use L2 when features are weakly informative, L1 when features are redundant
        {"units": 16, "activation": "relu"}#, 'kernel_regularizer': regularizers.l2(0.0001)} # Use L2 when features are weakly informative, L1 when features are redundant
    ],

    output_units=1,
    output_activation="sigmoid",
    padding="same",

    learning_rate=0.0005,
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision')],

    ####### Training the Model ##########
    epochs=30,
    batch_size=128
)

