import pandas as pd
import pandas_ta as ta
import numpy as np
import sys


class Data_Store():

    # This constructor is here to insure that when the class is invoked it returns a dataframe as an output
    def __new__(cls, data, feature, params, all_combinations=True, plot=False, char_type='line'):
        instance = super().__new__(cls)
        instance.__init__(data, feature, params, all_combinations, plot, char_type) # Invoke the class constructor with the given parametres
        return instance.feature_df  # Return the DataFrame directly



    # Initiate the class that will generate a all the combinations of parametres for a given feature
        # The features need to be passed as vectors i.e. length from 9 to 25 and type of price "high", 'low', 'close' etc....
    def __init__(self, data, feature, params, all_combinations = True, plot = False, char_type = 'line'):

        self.data = data # The data on which the feature will be calculated
        self.feature = feature # The name of the feature itself that will be calculated
        self.params = params # The Parametres that will be used to calculate the feature
        self.all_combinations = all_combinations # Whether we want to create features with all possible combinations of all combatible parametres
        self.feature_df = pd.DataFrame(index = self.data.index) # This will be the dataframe where all the features will be saved and returned as output
        self.plot = plot
        self.char_type = char_type


        self.params_2d_array = [] # This will hold all the feature vectors i.e. all each hyperparameter will be its own sublist 
        self.extract_features_into_lists() # Populate the above mentioned array

        self.parameter_combinations = [] # All the possible combinations of the different vectors will be stored here as an array of arrays
        self.generate_feature_combinations() # Populate the above mentioned array

        self.calculate_feature_combinations() # This calculates the feature together with all its combinations

    

    def extract_features_into_lists(self): # This will take the vectors of features and save them into an accessible list of lists i.e. if we have vector_1, vector_2 we get this into [vector_1, vector_2], where vectpr_1 and vector_2 are lists themselves

        for key, value in self.params.items():

            # Very often the feature is given as a range of values hence we need to unpack it
            if type(self.params[key]) == range: # Check it is a range class
                array = [i for i in self.params[key]] # Create a list holding all the values for a given parameter
                self.params_2d_array.append(array)
            
            elif (type(self.params[key]) == int) or (type(self.params[key]) == float): # Check if the parameter is just a number (int or float), then just create a list holding 1 value
                array = [self.params[key]] # Create a list holding only 1 integer or float
                self.params_2d_array.append(array)

            
            elif type(self.params[key]) == list: # Chck if the parameter is a list then we do not need to modify it further
                array = self.params[key] # Assign the list to a new variable
                self.params_2d_array.append(array)

            
            elif type(self.params[key]) == str: # Check if the parameter is a single string 
                array = [self.params[key]] # If so create a list with only the string value as an element
                self.params_2d_array.append(array)




    def generate_feature_combinations(self): # Create all the possible combinations of feature values across the different parameter vectors
        import itertools

        # Work in progress to implement functionality to take only the columns you want to combine
        if self.all_combinations == True:
            self.parameter_combinations = list(itertools.product(*self.params_2d_array)) # Get all the possible combinations by taking 1 element from each parameter vector
            # i.e. [[a,b,c], [d,e,f]] returns [(a,d), (a,e), (a,f), (b,d) etc .....]


    def calculate_feature_combinations(self): # Calculate all the combinations for a given feature
        for parameter_combination in self.parameter_combinations: # Loop over each combination

            parameter_combination = list(parameter_combination) # Since each combination is represented as a tuple turn it into a list
            
            parameter_names = [key for key,value in self.params.items()] # Extract the names for the hyperparemetres


            parameter_combination = {parameter_names[i]: parameter_combination[i] for i in range(0, len(parameter_names))} # Turn the combination into a dictionary

            # Here we are checking for the particular feature we are working with
            if self.feature == "ATR":
                # Try Caclulating the ATR if not we will raise an error showing that some of the mandatory features are missing
                try:
                    
                    # There is a possibility that some rudimentary parametres except the ones used for calculating the ATR are passed i.e. we have "length" and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of "length" with each value of "asd", however, this will be redundant as we will need only one value for length regardless of its combination with "asd" values
                    # In short this saves time
                    if f"ATR({parameter_combination['length']})" not in self.feature_df.columns:

                        feature = ta.atr(high = self.data['high'], low = self.data['low'], close = self.data['close'], length = parameter_combination['length']) # Calculate the ATR
                        self.feature_df[f"ATR({parameter_combination['length']})"] = feature # Save it into the output dataframe

                except:
                    raise ValueError(f"One of the required parametres /'high', 'low', 'close' for the data or 'length' for ATR is missing/ ")
                
            elif self.feature == "SMA":
                # Try Caclulating an SMA if not we will raise an error showing that some of the mandatory features are missing
                try:
                    
                    # There is a possibility that some rudimentary parametres except the ones used for calculating the SMA are passed i.e. we have "length" and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of "length" with each value of "asd", however, this will be redundant as we will need only one value for length regardless of its combination with "asd" values
                    # In short this saves time

                    # Set up that the default SMA will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'

                    if f"SMA({parameter_combination['length']}) {parameter_combination['source']}" not in self.feature_df.columns:

                        feature = ta.sma(close = self.data[parameter_combination['source']], length = parameter_combination['length']) # Calculate the SMA
                        self.feature_df[f"SMA({parameter_combination['length']}) {parameter_combination['source']}"] = feature # Save it into the output dataframe

                except:
                    raise ValueError(f"For SMA you need 'length' defined as int/float and if passing data make sure it is written as 'close' otherwise specify in 'source' which column to be used")
        

            elif self.feature == "EMA":
                # Try Caclulating an EMA if not we will raise an error showing that some of the mandatory features are missing
                try:
                    
                    # There is a possibility that some rudimentary parametres except the ones used for calculating the SMA are passed i.e. we have "length" and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of "length" with each value of "asd", however, this will be redundant as we will need only one value for length regardless of its combination with "asd" values
                    # In short this saves time

                    # Set up that the default EMA will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'


                    if f"EMA({parameter_combination['length']}) {parameter_combination['source']}" not in self.feature_df.columns:

                        feature = ta.ema(close = self.data[parameter_combination['source']], length = parameter_combination['length']) # Calculate the EMA
                        self.feature_df[f"EMA({parameter_combination['length']}) {parameter_combination['source']}"] = feature # Save it into the output dataframe

                except:
                    raise ValueError(f"For EMA you need 'length' defined as int/float and if passing data make sure it is written as 'close' otherwise specify in 'source' which column to be used")
        
            elif self.feature == "RSI":
                # Try Caclulating RSI if not we will raise an error showing that some of the mandatory features are missing
                try:


                    # Set up that the default RSI will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'

                    
                    # There is a possibility that some rudimentary parametres except the ones used for calculating the RSI are passed i.e. we have "length" and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of "length" with each value of "asd", however, this will be redundant as we will need only one value for length regardless of its combination with "asd" values
                    # In short this saves time

                    if f"RSI({parameter_combination['length']}) {parameter_combination['source']}" not in self.feature_df.columns:

                        feature = ta.rsi(close = self.data[parameter_combination['source']], length = parameter_combination['length'])
                        self.feature_df[f"RSI({parameter_combination['length']}) {parameter_combination['source']}"] = feature # Save it into the output dataframe

                except:
                    raise ValueError(f"For RSI you need 'length' defined as int/float and if passing data make sure it is written as 'close' otherwise specify in 'source' which column to be used, 'drift' is an optional parameter ")
        
            elif self.feature == 'KAMA': # Can't get it to work always returns None no matter what I do, will go back to it later

                # Try Caclulating KAMA if not we will raise an error showing that some of the mandatory features are missing
                try:

                    # Set up that the default KAMA will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'

                    # Enforce that the length is between the slow and fast thresholds
                    if parameter_combination['length'] < parameter_combination['fast'] or parameter_combination['length'] > parameter_combination['slow']:
                        raise ValueError(f"Length should be between 'fast' and 'slow', you have passed 'length': {parameter_combination['length']}, 'fast': {parameter_combination['fast']}, 'slow': {parameter_combination['slow']}")

                    # There is a possibility that some rudimentary parametres except the ones used for calculating the KAMA are passed i.e. we have "length", 'fast', 'slow' and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of 'length', 'fast' and 'slow' with  "asd", however, this will be redundant as we will need only one unique combination of 'length', 'fast', 'slow' regardless of "asd" values
                    # In short this saves time


                    if f"KAMA({parameter_combination['length']}) Fast {parameter_combination['fast']} Slow {parameter_combination['slow']}" not in self.feature_df.columns:
                        
                        feature = ta.kama(close = self.data[parameter_combination['source']], length = parameter_combination['length'], fast = parameter_combination['fast'], slow = parameter_combination['slow'])
                        self.feature_df[f"KAMA({parameter_combination['length']}) Fast {parameter_combination['fast']} Slow {parameter_combination['slow']}"] = feature # Save it into the output dataframe

                except:
                    raise ValueError(f"KAMA requires 3 mandatory arguements 'length', 'fast', 'slow' in int format, make sure the data has a 'close' column otherise manually set a 'source' ")
        
            elif self.feature == "MACD":
                # Try Caclulating MACD if not we will raise an error showing that some of the mandatory features are missing
                try:

                    # Set up that the default MACD will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'

                    # Enforce that the length is between the slow and fast thresholds
                    #if parameter_combination['signal'] < parameter_combination['fast'] or parameter_combination['signal'] > parameter_combination['slow']:
                    #    raise ValueError(f"Signal should be between 'fast' and 'slow', you have passed 'signal': {parameter_combination['signal']}, 'fast': {parameter_combination['fast']}, 'slow': {parameter_combination['slow']}")

                    # There is a possibility that some rudimentary parametres except the ones used for calculating the MACD are passed i.e. we have "signal", 'fast', 'slow' and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of 'signal', 'fast' and 'slow' with  "asd", however, this will be redundant as we will need only one unique combination of 'signal', 'fast', 'slow' regardless of "asd" values
                    # In short this saves time

                    if f"MACD({parameter_combination['signal']}) Fast {parameter_combination['fast']} Slow {parameter_combination['slow']}" not in self.feature_df.columns:
                        feature = ta.macd(close = self.data[parameter_combination['source']], signal = parameter_combination['signal'], fast = parameter_combination['fast'], slow = parameter_combination['slow'])

                        self.feature_df[f"MACD({parameter_combination['signal']}) Fast {parameter_combination['fast']} Slow {parameter_combination['slow']}"] = feature.iloc[:,0] # Saves the MACD
                        self.feature_df[f"MACD Signal({parameter_combination['signal']}) Fast {parameter_combination['fast']} Slow {parameter_combination['slow']}"] = feature.iloc[:,2] # Saves the Signal Line

                except:
                    raise ValueError(f"MACD requires 3 mandatory arguements 'signal', 'fast', 'slow' in int format, make sure the data has a 'close' column otherise manually set a 'source' ")
        
            elif self.feature == "Stoch":

                # Try Caclulating Stochastic Momentum if not we will raise an error showing that some of the mandatory features are missing
                try:


                    # There is a possibility that some rudimentary parametres except the ones used for calculating the Stochastic are passed i.e. we have "k", 'd', 'smooth_k' and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of 'k', 'd' and 'smooth_k' with  "asd", however, this will be redundant as we will need only one unique combination of 'k', 'd', 'smooth_k' regardless of "asd" values
                    # In short this saves time

                    if f"Stoch({parameter_combination['k']}) {parameter_combination['d']} {parameter_combination['smooth_k']} K" not in self.feature_df.columns:
                        feature = ta.stoch(high = self.data['high'], low = self.data['low'], close = self.data['close'], k = parameter_combination['k'], d = parameter_combination['d'], smooth_k= parameter_combination['smooth_k'])
                        self.feature_df[f"Stoch({parameter_combination['k']}) {parameter_combination['d']} {parameter_combination['smooth_k']} K"] = feature.iloc[:,0]
                        self.feature_df[f"Stoch({parameter_combination['k']}) {parameter_combination['d']} {parameter_combination['smooth_k']} D"] = feature.iloc[:,1]

                except:
                    raise ValueError(f"Stochastic requires 3 mandatory arguements 'k', 'd', 'smooth_k' in int format, the indicator is hardcoded to expect 'high', 'low' and 'close' for data sources")
        
            elif self.feature == "Hawkes":
                from hawkes import hawkes_process

                    # Try Caclulating the Hawkes Process if not we will raise an error showing that some of the mandatory features are missing
                try:

                    # There is a possibility that some rudimentary parametres except the ones used for calculating the Hawkes are passed i.e. we have "kappa", 'lookback' and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of 'kappa', 'lookback'  with  "asd", however, this will be redundant as we will need only one unique combination of 'kappa', 'loookback' regardless of "asd" values
                    # In short this saves time

                    if f"Hawkes({parameter_combination['kappa']}) {parameter_combination['lookback']}" not in self.feature_df.columns:

                        atr = ta.atr(np.log(self.data['high']), np.log(self.data['low']), np.log(self.data['close']), parameter_combination['lookback'])
                        norm_range = ( np.log(self.data['high']) - np.log(self.data['low']) ) / atr
                        feature = hawkes_process(data = norm_range, kappa = parameter_combination['kappa'] )
                        self.feature_df[f"Hawkes({parameter_combination['kappa']}) {parameter_combination['lookback']}"] = feature


                except:
                    raise ValueError(f"Hawkes requires 2 mandatory arguements 'lookback' in int format and 'kappa' in float format, the indicator is hardcoded to expect 'high', 'low' and 'close' for data sources")
        
            elif self.feature == "Reversability": # Warning can be slow to calculate do not overload with a lot of options of 'lookback', espeically long ones, though the minimum is around 60

                from reversability import rw_ptsr

                    # Try Caclulating the price reversability index using ordinal patters. It ranges from 0 to 1, 0 indicates mean reversion, 1 very strong trend
                try:

                    # Set up that the default Reversability index will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                    if 'source' not in parameter_combination.keys():
                        parameter_combination['source'] = 'close'

                    # There is a possibility that some rudimentary parametres except the ones used for calculating the Reversability Index are passed i.e. we have 'lookback' and "asd" hyperparemetres
                    # Clearly the script might make combinations of each value of 'lookback' with  "asd", however, this will be redundant as we will need only 'loookback' regardless of "asd" values
                    # In short this saves time

                    if f"Reversability({parameter_combination['lookback']}) {parameter_combination['source']}" not in self.feature_df.columns:

                        feature = rw_ptsr(arr = self.data[parameter_combination['source']].to_numpy(), lookback = parameter_combination['lookback'] )
                        self.feature_df[f"Reversability({parameter_combination['lookback']}) {parameter_combination['source']}"] = feature


                except:
                    raise ValueError(f"The Reversability Ordinal Pattern Feature requires 1 mandatory arguement: 'lookback' in int format, the data is expected to have a 'close' column if not pass an arguement 'source' to define where data needs to be pulled from")
        

            elif self.feature == 'Return':
                    # Try Caclulating MACD if not we will raise an error showing that some of the mandatory features are missing
                    try:

                        # Set up that the default MACD will be calculated on close prices i.e. we do not have to pass it as a hyperparameter by default
                        if 'source' not in parameter_combination.keys():
                            parameter_combination['source'] = 'close'


                        # There is a possibility that some rudimentary parametres except the ones used for calculating the retur are passed i.e. we have "signal", 'period' and "asd" hyperparemetres
                        # Clearly the script might make combinations of each value of 'signal', 'fast' and 'slow' with  "asd", however, this will be redundant as we will need only one unique combination of 'signal', 'fast', 'slow' regardless of "asd" values
                        # In short this saves time


                        if f"Return({parameter_combination['length']})" not in self.feature_df.columns:
                            feature = ta.momentum.roc(self.data[parameter_combination['source']], window  = parameter_combination['length'])

                            self.feature_df[f"Return({parameter_combination['length']})"] = feature
                    except:
                        raise ValueError(
                            f"Return requires 1 mandatory arguements 'length' in int format, make sure the data has a 'close' column otherise manually set a 'source' ")


if __name__ == "__main__":
    data = pd.read_csv("BTCUSDT3600.csv")

    features = Data_Store(data, 'SMA', {"lookback": [40, 60, 80]} )


    print(features)

