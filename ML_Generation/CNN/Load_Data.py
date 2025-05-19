import pandas as pd
import sys

def Load_Data_From_CSV(data, t=1, t_n=5):
    # Load the CSV file
    #filename = f'{data}.csv'
    #data = pd.read_csv(data, parse_dates=['Date'], index_col='Date')

    # Ensure 'Close' column exists(
    if ('Close' not in data.columns) and ('close' not in data.columns):
        raise ValueError("The CSV file must contain a 'Close' or a 'close' column i.e. choose one convention and provide OHLCV data in upper or lower case")
    # Check the data types of the columns
    # Create labels based on future price movements
    try:
        return_ = (data['Close'].shift(-t_n) - data['Open'].shift(-t) ) / data['Open'].shift(-t)
        data['Label'] = ( (return_ > - 0.01) & (return_ < 0.01) ).astype(int)
        #data['Label'] = (return_ < -0.01).astype(int)

        data['1-Period Forward Return'] = (data['Close'].shift(-1) - data['Close'] ) / data['Close']
        data['Return'] = return_
    
    except:
        return_ = (data['close'].shift(-t_n) - data['open'].shift(-t) ) / data['open'].shift(-t)
        data['Label'] = ( (return_ > - 0.01) & (return_ < 0.01) ).astype(int)
        #data['Label'] = (return_ < -0.01).astype(int)

        data['1-Period Forward Return'] = (data['close'].shift(-1) - data['close'] ) / data['close']
        data['Return'] = return_
    
    print("Label distribution:\n", data['Label'].value_counts())


    # Save the last 10% of rows for prediction and add self.t_n rows for an offset, this will be our out of sample data
    test_data = data[int(0.9 * len(data)) + t_n:].copy()
    test_data = test_data[: - t_n]

    #unknown_data.dropna(inplace = True)

    # Take the first 90% as known data, this will be the training and validation data
    training_validation_data = data.iloc[:int(0.9 * len(data))].copy()

    return training_validation_data, test_data

# Optional: test run
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    Data = pd.read_csv("BTCUSDT3600.csv")
    Known_df, Unknown_df = Load_Data_From_CSV(Data, t=1, t_n=5)

    print("First few rows of known data:")
    print(Known_df.head())

    print("First few rows of unknown data:")
    print(Unknown_df)
