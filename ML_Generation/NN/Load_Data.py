import pandas as pd

def Load_Data_From_CSV(data, t=1, t_n=5):
    # Load the CSV file
    #filename = f'{data}.csv'
    #data = pd.read_csv(data, parse_dates=['Date'], index_col='Date')

    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        raise ValueError("The CSV file must contain a 'Close' column.")
    # Check the data types of the columns
    print(data.dtypes)
    # Create labels based on future price movements
    data['Label'] = ( (data['Close'].shift(-t_n) - data['Open'].shift(-t) ) / data['Open'].shift(-t)
                           < -0.005 ).astype(int)
    print("Label distribution:\n", data['Label'].value_counts())
    print(data[['Close']].head(10))
    print("Shifted Close (t):", data['Close'].shift(-1).head(10))
    print("Shifted Close (t_n):", data['Close'].shift(-5).head(10))
    # Save the last t_n rows for prediction
    unknown_data = data.iloc[-t_n:].copy()

    # Exclude the last t_n rows for training/testing
    known_data = data.iloc[:-t_n].copy()

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
