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

# Import Customizable Neural Network
from Customizable_NN import create_model


class NN_Classifier:
    def __init__(self,
    #Dataset
    Data = 'BTCUSDT3600',
    t_n = 5,
    ####### Creating the Model #########
    layer_sizes=np.arange(10, 30, 5),
    activation='relu',
    output_activation='sigmoid',
    output_units = 1,
    dropout_mode='uniform',
    dropout=0.2,
    learning_rate=0.0005,
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision')],
    ####### Training the Model ##########
    epochs=30,
    batch_size=64
    ):
        #os.evniron['TF_ENABLE_ONEDNN_OPTS'] = "0"
        self.Data = Data
        self.t_n = t_n
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        ####### Creating the Model #########
        self.layer_sizes=layer_sizes
        self.activation=activation
        self.output_activation=output_activation
        self.output_units = output_units
        self.dropout_mode=dropout_mode
        self.dropout=dropout
        self.learning_rate = learning_rate
        self.optimizer=optimizer
        self.loss=loss
        self.metrics=metrics
        ####### Training the Model ##########
        self.epochs=epochs
        self.batch_size=batch_size

        self.load_and_prepare_data()

        self.build_model()

        self.train()

        self.evaluate()  # No inputs
        self.predict_unknown()  # No inputs
        self.plot_Confusion_Matrix()  # No inputs
        self.plot_Accuracy_Precision()  # No inputs

    def load_and_prepare_data(self):
        Known_Data, Unknown_Data = Load_Data_From_CSV(self.Data,
                                                      t=1,
                                                      t_n = self.t_n)
        self.Unknown_Data = Unknown_Data

        Features = Known_Data.drop(columns=['Label', 'Date']) #Sanity Check Known_Data[['Label']] vs Known_Data.drop(columns='Label')
        Labels = Known_Data['Label']

        self.Features_Train, self.Features_Test, self.Labels_Train, self.Labels_Test = train_test_split(Features,
                                                                                                        Labels,
                                                                                                        test_size=0.3,
                                                                                                        shuffle=False
                                                                                                        )
        ################# For Futre Make Sure to Scale only based On data up to t0 to avoid future data leakage#######
        self.Features_Train = self.scaler.fit_transform(self.Features_Train)
        self.Features_Test = self.scaler.transform(self.Features_Test)

    def build_model(self):

        Input_Dim = self.Features_Train.shape[1]
        self.model = create_model(
                 input_shape = Input_Dim,
                 hidden_layer_sizes = self.layer_sizes,
                 hidden_layer_activations=self.activation,
                 output_units=self.output_units,
                 output_activation=self.output_activation,
                 dropout_mode=self.dropout_mode,
                 dropout_value=self.dropout,
                 learning_rate=self.learning_rate
                 )

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics
                           )
        self.model.summary()

    def train(self):
        self.history = self.model.fit(
            self.Features_Train,
            self.Labels_Train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.Features_Test, self.Labels_Test)
        )

    def evaluate(self):
        loss, acc, precision = self.model.evaluate(self.Features_Test,
                                                   self.Labels_Test,
                                                   self.batch_size)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Precision: {precision:.4f}")

        preds = (self.model.predict(self.Features_Test) > 0.5).astype(int)
        print(confusion_matrix(self.Labels_Test, preds))
        print(classification_report(self.Labels_Test, preds))

    def predict_unknown(self):
        ################# For Futre Make Sure to Scale only based On data up to t0 to avoid future data leakage#######
        unknown_scaled = self.scaler.transform(self.Unknown_Data.drop(columns=['Label', 'Date']))
        predictions = (self.model.predict(unknown_scaled) > 0.5).astype(int)
        print(predictions)
        return predictions

    def plot_Confusion_Matrix(self):
        preds = (self.model.predict(self.Features_Test) > 0.5).astype(int)
        cm = confusion_matrix(self.Labels_Test, preds)
        TN, FP, FN, TP = cm.ravel()

        TPR_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        FNR_1 = FN / (TP + FN) if (TP + FN) > 0 else 0
        TNR_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        FPR_0 = FP / (TN + FP) if (TN + FP) > 0 else 0

        PPV_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        FDR_1 = FP / (TP + FP) if (TP + FP) > 0 else 0
        NPV_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        FOR_0 = FN / (TN + FN) if (TN + FN) > 0 else 0

        fpr, tpr, _ = roc_curve(self.Labels_Test, self.model.predict(self.Features_Test))
        fpr_0, tpr_0, _ = roc_curve(1 - self.Labels_Test, self.model.predict(self.Features_Test))
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
    NN_Classifier(
        #Dataset
        Data = 'BTCUSDT3600',
        t_n = 5,
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

    '''clf.load_and_prepare_data()
    
    clf.build_model(layer_sizes=np.arange(10, 30, 5),
                        activation='relu',
                        output_activation='sigmoid',
                        dropout_mode='uniform',
                        dropout=0.2,
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy', Precision(name='precision')]
                        )
    
    clf.train(epochs=30,
              batch_size=64
              )
    
    clf.evaluate() #No inputs
    clf.predict_unknown() #No inputs
    clf.plot_Confusion_Matrix() #No inputs
    clf.plot_Accuracy_Precision() #No inputs'''