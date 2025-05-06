# Import os
import os
import sys

# Import Pandas
import pandas as pd

# Import Numpy
import numpy as np

# Custom Functions
from feature_generator import Data_Store


class Input_Data_Organizer:
    def __init__(self,
                 

                 #Data Used to Calculate Indicators
                 Data = 'BTCUSDT3600.csv',

                 #Simple Moving Average--------------------
                 Use_SMA = False,
                 #Choose Length or Lengths for SMA
                 SMA_Length = list(np.arange(5, 50, 5)),
                 #Choose What to Column to use to Calculate SMA on
                 SMA_Source = 'Close',

                 # Simple Moving Average--------------------
                 Use_EMA = False,
                 # Choose Length or Lengths for SMA
                 EMA_Length = list(np.arange(5, 50, 5)),
                 # Choose What to Column to use to Calculate SMA on
                 EMA_Source = 'Close',

                 # Returns--------------------
                 Use_Return=False,
                 # Choose Length or Lengths for Return
                 Return_Length=list(np.arange(5, 50, 5)),
                 # Choose What to Column to use to Calculate Return on
                 Return_Source='Close',

                 # RSI--------------------
                 Use_RSI=False,
                 # Choose Length or Lengths for RSI
                 RSI_Length=14,
                 # Choose What to Column to use to Calculate RSI on
                 RSI_Source='Close',

                 # MACD--------------------
                 Use_MACD=False,
                 MACD_Signal = 9,
                 MACD_Fast = 12,
                 MACD_Slow = 26,
                 MACD_Source='Close',

                 # Hawkes
                 Use_Hawkes = False,
                 Hawkes_Kappa = 0.1,
                 Hawkes_Lookback = 168,
                 Hawkes_Source = ['High', 'Low', 'Close'],

                 # Reversability
                 Use_Reversability = False,
                 Reversability_Lookback = 60, # Do at least 60 otherwise the calculations might not be stable 
                 Reversability_Source = 'Close'               
                 ):



        self.Data = pd.read_csv(Data)
        self.Data["Date"] = pd.to_datetime(self.Data["Date"])

        #SMA ----------------------------------
        self.Use_SMA = Use_SMA
        self.SMA_Length = SMA_Length
        self.SMA_Source = SMA_Source

        #EMA ----------------------------------
        self.Use_EMA = Use_EMA
        self.EMA_Length = EMA_Length
        self.EMA_Source = EMA_Source

        #Return ----------------------------------
        self.Use_Return = Use_Return
        self.Return_Length = Return_Length
        self.Return_Source = Return_Source

        #RSI ----------------------------------
        self.Use_RSI= Use_RSI
        self.RSI_Length = RSI_Length
        self.RSI_Source = RSI_Source

        # MACD ----------------------------------
        self.Use_MACD = Use_MACD
        self.MACD_Signal = MACD_Signal
        self.MACD_Slow = MACD_Slow
        self.MACD_Fast = MACD_Fast
        self.MACD_Source = MACD_Source

        # Hawkes ----------------------------------
        self.Use_Hawkes = Use_Hawkes
        self.Hawkes_Kappa = Hawkes_Kappa
        self.Hawkes_Lookback = Hawkes_Lookback
        self.Hawkes_Source = Hawkes_Source

        # Reversability ----------------------------------
        self.Use_Reversability = Use_Reversability
        self.Reversability_Lookback = Reversability_Lookback
        self.Reversability_Source = Reversability_Source


        if self.Use_SMA:
            self.Calculate_SMAs()

        if self.Use_EMA:
            self.Calculate_EMAs()

        if self.Use_Return:
            self.Calculate_Return()

        if self.Use_RSI:
            self.Calculate_RSI()

        if self.Use_MACD:
            self.Calculate_MACD()

        if self.Use_Hawkes:
            self.Calculate_Hawkes()

        if self.Use_Reversability:
            self.Calculate_Reversability()

        # Drop initial rows with NaN
        self.Drop_NaN_Rows()


    def Calculate_SMAs(self):
        self.SMAs = Data_Store(self.Data,
                               'SMA',
                               {"length": self.SMA_Length,
                                "source": self.SMA_Source}
                               )
        self.Data[self.SMAs.columns] = self.SMAs
        return self.Data

    def Calculate_EMAs(self):
        self.EMAs = Data_Store(self.Data,
                               'EMA',
                               {"length": self.EMA_Length,
                                "source": self.EMA_Source}
                               )
        self.Data[self.EMAs.columns] = self.EMAs
        return self.Data

    def Calculate_Return(self):
        self.Return = Data_Store(self.Data,
                                 'Return',
                                 {'length': self.Return_Length,
                                  'source': self.Return_Source})
        self.Data[self.Return.columns] = self.Return
        return self.Data

    def Calculate_RSI(self):

        self.RSI = Data_Store(self.Data,
                                 'RSI',
                                 params = {'length': self.RSI_Length,
                                  'source': self.RSI_Source})
        self.Data[self.RSI.columns] = self.RSI
        return self.Data
    
    def Calculate_MACD(self):
        self.MACD = Data_Store(self.Data,
                                 'MACD',
                                 {'signal': self.MACD_Signal,
                                  'slow': self.MACD_Slow,
                                  'fast': self.MACD_Fast,
                                  'source': self.MACD_Source})
        self.Data[self.MACD.columns] = self.MACD
        return self.Data


    def Calculate_Hawkes(self):


        if 'close' not in self.Data.columns:
            self.Data['high'] = self.Data[self.Hawkes_Source[0]]
            self.Data['low'] = self.Data[self.Hawkes_Source[1]]
            self.Data['close'] = self.Data[self.Hawkes_Source[2]]



            self.Hawkes= Data_Store(self.Data,
                                    'Hawkes',
                                    params = {'kappa': self.Hawkes_Kappa,
                                    'lookback': self.Hawkes_Lookback
                                    })
            self.Data[self.Hawkes.columns] = self.Hawkes

            self.Data = self.Data.drop(['high', 'low', 'close'], axis = 1)
        
        else:
            self.Hawkes= Data_Store(self.Data,
                                    'Hawkes',
                                    params = {'kappa': self.Hawkes_Kappa,
                                    'lookback': self.Hawkes_Lookback
                                    })
            self.Data[self.Hawkes.columns] = self.Hawkes
        return self.Data
    
    def Calculate_Reversability(self):
        self.Reversability = Data_Store(self.Data,
                                        "Reversability",
                                        {'lookback': self.Reversability_Lookback,
                                         'source': self.Reversability_Source})
        self.Data[self.Reversability.columns] = self.Reversability
        return self.Data

    def Drop_NaN_Rows(self):
        #max_sma = max(self.SMA_Length) if self.Use_SMA else 0
        #max_ema = max(self.EMA_Length) if self.Use_EMA else 0
        #max_length = max(max_sma, max_ema)

        self.Data.dropna(inplace = True)
        self.Data = self.Data.reset_index(drop=True)

    def Return_Data(self):
        return self.Data


if __name__ == "__main__":
    # Example usage with np.arange
    Organizer = Input_Data_Organizer(
        # Data Used to Calculate Indicators
        Data='BTCUSDT3600.csv',

        # Simple Moving Average--------------------
        Use_SMA=True,
        # Choose Length or Lengths for SMA
        SMA_Length=list(np.arange(9, 25, 1)),
        # Choose What to Column to use to Calculate SMA on
        SMA_Source='Close',

        # Simple Moving Average--------------------
        Use_EMA=True,
        # Choose Length or Lengths for SMA
        EMA_Length=list(np.arange(9, 25, 1)),
        # Choose What to Column to use to Calculate SMA on
        EMA_Source='Close'
    )
    df = Organizer.Return_Data()