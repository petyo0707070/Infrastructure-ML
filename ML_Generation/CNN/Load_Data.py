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
        data['Label'] = ( (data['Close'].shift(-t_n) - data['Open'].shift(-t) ) / data['Open'].shift(-t) < -0.00 ).astype(int)
        data['Return'] = ( (data['Close'].shift(-t_n) - data['Open'].shift(-t) ) / data['Open'].shift(-t) )
    
    except:
        data['Label'] = ( (data['close'].shift(-t_n) - data['open'].shift(-t) ) / data['open'].shift(-t) < -0.00 ).astype(int)
        data['Return'] = ( (data['close'].shift(-t_n) - data['open'].shift(-t) ) / data['open'].shift(-t) )
    
    print("Label distribution:\n", data['Label'].value_counts())


    # Save the last 10% of rows for prediction and add 2 rows for embargo, this will be our out of sample data
    unknown_data = data[int(0.9 * len(data)) + 2:].copy()
    unknown_data = unknown_data[: - t_n]

    #unknown_data.dropna(inplace = True)

    # Take the first 90% as known data, this will be the training and validation data
    known_data = data.iloc[:int(0.9 * len(data))].copy()

    return known_data, unknown_data

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
