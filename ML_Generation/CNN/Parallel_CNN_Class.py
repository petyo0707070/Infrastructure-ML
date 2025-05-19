################################################## Import Libraries ####################################################
# Import tensorflow
import sys

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

# Import seaborn
import seaborn as sns  # For plotting the confusion matrices

# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np

# Import Matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import Data
from Load_Data import Load_Data_From_CSV

# Import Customizable Parallel CNN Model
from Customizable_Parallel_CNN import build_parallel_cnn_model  # Import the new CNN function


class Parallel_CNN_Classifier:
    def __init__(self,
                 # Dataset
                 Data='BTCUSDT3600',
                 t_n=5,
                 include_ohlcv = False,
                 predict_return_positive = True,

                 ####### Creating the Model #########
                 kernel_depth = 60, # The max lookback that the kernel would be able to go 
                 conv_branches=[{"filters": 64, "kernel_size": 3}], # Notice the difference with kernel_size which is how many rows at once does the kernel analyze
                 pool_type="max",  # "max" or "avg"
                 pool_size=2,
                 dropout_rate=0.3,
                 dense_layer_dropout_rate = 0.3,
                 dense_layers=[{"units": 128, "activation": "relu"}],
                 output_units=1,
                 output_activation="sigmoid",
                 padding="same",  # User chooses whether to use padding
                 learning_rate=0.0005,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', Precision(name='precision')],
                 ####### Training the Model ##########
                 epochs=30,
                 batch_size=64
                 ):
        
        # Assert that  the kernel depth is at least as big as the biggest kernel_size
        max_kernel_size = 0

        for y in conv_branches:
            for x in y:
                if x['kernel_size'] > max_kernel_size:
                    max_kernel_size = x['kernel_size']

        assert(kernel_depth > max_kernel_size)
        self.Data = Data
        self.t_n = t_n
        self.include_ohlcv = include_ohlcv
        self.predict_return_positive = predict_return_positive
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        ####### Creating the Model #########
        self.kernel_depth = kernel_depth
        self.conv_branches = conv_branches
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.dense_layer_dropout_rate = dense_layer_dropout_rate
        self.dense_layers = dense_layers
        self.output_units = output_units
        self.output_activation = output_activation
        self.padding = padding
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        ####### Training the Model ##########
        self.epochs = epochs
        self.batch_size = batch_size

        self.load_and_prepare_data()


        self.Features_Train_reshaped = self.create_sliding_windows(data_1 = self.Features_Train)
        self.Features_Validation_reshaped = self.create_sliding_windows(data_1= self.Features_Validation)
        self.Features_Test_reshaped = self.create_sliding_windows(data_1 = self.Features_Test)
        self.build_model()

        self.train()

        self.evaluate()  # No inputs
        self.plot_Confusion_Matrix()  # No inputs
        self.plot_Accuracy_Precision()  # No inputs

    def load_and_prepare_data(self):
        training_validation_data, test_data = Load_Data_From_CSV(self.Data,
                                                      t=1,
                                                      t_n=self.t_n)
        
        # The label offset is already implemented in Load_Data for the test set
        self.Features_Test = test_data.drop(columns = ['Label', 'Date', 'Return', '1-Period Forward Return'])
        self.Labels_Test = test_data[['Label', 'Return', '1-Period Forward Return']]

        if self.include_ohlcv:
            Features = training_validation_data.drop(columns=['Label', 'Date', 'Return', '1-Period Forward Return'])  # Sanity Check Known_Data[['Label']] vs Known_Data.drop(columns='Label')
        else:
            try:
                try:
                    Features = training_validation_data.drop(columns=['Label', 'Date', 'Return', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    self.Features_Test = self.Features_Test.drop( columns = ['Open', 'High', 'Low', 'Close', 'Volume'])
                except:
                    Features = training_validation_data.drop(columns=['Label', 'Date', 'Return', 'open', 'high', 'low', 'close', 'volume'])
                    self.Features_Test = self.Features_Test.drop( columns = ['open', 'high', 'low', 'close', 'volume'])

            except:
                raise ValueError("Either include columns written as OHLCV or ohlcv i.e. 'Close' or 'close', you need all 5 columns present")

        Labels = training_validation_data[['Label', 'Return', '1-Period Forward Return']]

        self.Features_Train, self.Features_Validation, self.Labels_Train, self.Labels_Validation = train_test_split(Features,
                                                                                                        Labels,
                                                                                                        test_size=0.2,
                                                                                                        shuffle=False
                                                                                                        )
        
        self.Features_Validation = self.Features_Validation[self.t_n:]
        self.Labels_Validation = self.Labels_Validation[self.t_n:]


        ################# For Future Make Sure to Scale only Based On data up to t0 to avoid future data leakage#######
        self.Features_Train = self.scaler.fit_transform(self.Features_Train)
        self.Features_Validation = self.scaler.transform(self.Features_Validation)
        self.Features_Test = self.scaler.transform(self.Features_Test)

        #print(self.Features_Train.shape)


    def create_sliding_windows(self, data_1):
        X = []
        for i in range(data_1.shape[0] - self.kernel_depth + 1):
            X.append(data_1[i:i + self.kernel_depth])

        return np.array(X)

    def build_model(self):
        input_shape = (self.kernel_depth, self.Features_Train.shape[1])  # Here we define the data that each kernel will be shown

        # Use the build_parallel_cnn_model function
        self.model = build_parallel_cnn_model(
            input_shape=input_shape,
            conv_branches=self.conv_branches,
            pool_type=self.pool_type,
            pool_size=self.pool_size,
            dropout_rate=self.dropout_rate,
            dense_layer_dropout_rate = self.dense_layer_dropout_rate,
            dense_layers=self.dense_layers,
            output_units=self.output_units,
            output_activation=self.output_activation,
            padding=self.padding
        )

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics
                           )
        self.model.summary()

    def train(self):


        self.history = self.model.fit(
            self.Features_Train_reshaped,
            self.Labels_Train[self.kernel_depth - 1:]['Label'],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.Features_Validation_reshaped, self.Labels_Validation[self.kernel_depth - 1:]['Label'])
        )

    def evaluate(self):
        loss, acc, precision = self.model.evaluate(self.Features_Test_reshaped,
                                                   self.Labels_Test[self.kernel_depth - 1:]['Label'],
                                                   self.batch_size)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")

        preds = (self.model.predict(self.Features_Test_reshaped) > 0.5).astype(int)
        in_position = False
        entry_i = 0
        equity_curve = []

        #for i in range(0, len(preds)):

        #    if in_position == False and preds[i] == 1:
        #        entry_i = i
        #        in_position = True

        #        if self.predict_return_positive:
        #            equity_curve.append(self.Labels_Test.iloc[i]['Return'])
        #        else:
        #            equity_curve.append(-1 * self.Labels_Test.iloc[i]['Return'])
            
        #    if in_position == True and i < entry_i + self.t_n:
        #        equity_curve.append(0)
            
        #    if in_position == True and i >= entry_i + self.t_n:
        #        in_position = False 
        #        entry_i = 0

# Implementation of an equity curve which continiously updates the postion, goes from a position size of -1 to 1 and changes by 10% of position size per prediction
# Currently testing the idea that continiously enter into the position based on prediction, but we close continiously after t_n canldes, essentially mimicking the entry logic

        position_entries = {int(i) : 0 for i in range(0, len(preds))}
        position_multiplyer = 0
        multiplyer_step = 0.1


        for i in range(0, len(preds)): # Loop over each candle

            # First examine the case if we are interested in longs
            if self.predict_return_positive == 1:

                if preds[i] == 1: # If it signals to go long

                    if position_multiplyer < 1: # Enter the position and hence only keep in mind the exits which are applicable if we were not in the full position
                        position_multiplyer += multiplyer_step
                        position_entries[i] = 1
                
                else: # For now this is a question, of whether we should do anything if we get a print of 0 for now we skip
                    pass

                if i >= self.t_n:  # We start checking for set-ups that have played out only after t_n candles have passed since we started counting
                    if position_entries[i - self.t_n] == 1: # Check if t_n candles have passed since entry
                        position_multiplyer += - multiplyer_step


            # This code snippet takes care of the short side of the trades
            if self.predict_return_positive == 0:

                if preds[i] == 1: # If it signals to go short

                    if position_multiplyer > -1: # Enter the position and hence only keep in mind the exits which are applicable if we were not in the full position
                        position_multiplyer += -multiplyer_step
                        position_entries[i] = 1


                    else: # For now this is a question, of whether we should do anything if we get a print of 0 for now we skip
                        pass

                    if i >= self.t_n:  # We start checking for set-ups that have played out only after t_n candles have passed since we started counting
                        if position_entries[i - self.t_n] == 1: # Check if t_n candles have passed since entry
                            position_multiplyer += multiplyer_step

            equity_curve.append(position_multiplyer * self.Labels_Test.iloc[i]['1-Period Forward Return'])

        plt.plot(np.cumsum(equity_curve))
        plt.title("Equity Curve on the Test Set")
        plt.show()               


        print(confusion_matrix(self.Labels_Test[self.kernel_depth - 1:]['Label'], preds))
        print(classification_report(self.Labels_Test[self.kernel_depth - 1:]['Label'], preds))



    def plot_Confusion_Matrix(self):
        preds = (self.model.predict(self.Features_Test_reshaped) > 0.5).astype(int)
        cm = confusion_matrix(self.Labels_Test[self.kernel_depth - 1:]['Label'], preds)
        TN, FP, FN, TP = cm.ravel()

        TPR_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        FNR_1 = FN / (TP + FN) if (TP + FN) > 0 else 0
        TNR_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        FPR_0 = FP / (TN + FP) if (TN + FP) > 0 else 0

        PPV_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        FDR_1 = FP / (TP + FP) if (TP + FP) > 0 else 0
        NPV_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        FOR_0 = FN / (TN + FN) if (TN + FN) > 0 else 0

        fpr, tpr, _ = roc_curve(self.Labels_Test[self.kernel_depth - 1:]['Label'], self.model.predict(self.Features_Test_reshaped))
        fpr_0, tpr_0, _ = roc_curve(1 - self.Labels_Test[self.kernel_depth - 1:]['Label'], self.model.predict(self.Features_Test_reshaped))
        auc_1 = auc(fpr, tpr)
        auc_0 = auc(fpr_0, tpr_0)

        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap([[TNR_0, FPR_0], [TPR_1, FNR_1]],
                    annot=[[f"{TNR_0:.2%}", f"{FPR_0:.2%}"], [f"{TPR_1:.2%}", f"{FNR_1:.2%}"]],
                    fmt='', cmap='Blues',
                    xticklabels=['TPR', 'FNR'], yticklabels=['Class 0', 'Class 1'], cbar=False, ax=ax2)
        ax2.set_title('Recall & Miss Rate')

        ax3 = fig.add_subplot(gs[1, 0])
        sns.heatmap([[NPV_0, PPV_1], [FOR_0, FDR_1]],
                    annot=[[f"{NPV_0:.2%}", f"{PPV_1:.2%}"], [f"{FOR_0:.2%}", f"{FDR_1:.2%}"]],
                    fmt='', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'], yticklabels=['Precision', 'Error Rate'], cbar=False, ax=ax3)
        ax3.set_title('Precision & Errors')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(fpr, tpr, color='orange', label=f'ROC Class 1 (AUC = {auc_1:.2f})')
        ax4.plot(fpr_0, tpr_0, color='blue', label=f'ROC Class 0 (AUC = {auc_0:.2f})')
        ax4.plot([0, 1], [0, 1], 'k--')
        ax4.set_title('ROC Curves')
        ax4.set_xlabel('FPR')
        ax4.set_ylabel('TPR')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def plot_Accuracy_Precision(self):
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Loss Over Epochs')
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Accuracy Over Epochs')
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.history.history['precision'], label='Train Precision')
        ax3.plot(self.history.history['val_precision'], label='Val Precision')
        ax3.set_title('Precision Over Epochs')
        ax3.legend()

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')  # Placeholder

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    Data = pd.read_csv("BTCUSDT3600_First_1k.csv")
    #Data = Data[:1000]
    Parallel_CNN_Classifier(
        # Dataset
        Data='Data',
        t_n=5,
        ####### Creating the Model #########
        conv_branches=[{"filters": 64, "kernel_size": 3}],
        pool_type="max",  # "max" or "avg"
        pool_size=2,
        dropout_rate=0.3,
        dense_layers=[{"units": 128, "activation": "relu"}],
        output_units=1,
        output_activation="sigmoid",
        padding="same",  # User chooses whether to use padding
        learning_rate=0.0005,
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision')],
        ####### Training the Model ##########
        epochs=30,
        batch_size=64
    )
