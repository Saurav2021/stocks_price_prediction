
Importing Necessary Libraries
python
Copy code
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
yfinance: Library for downloading stock data from Yahoo Finance.
pandas: Library for data manipulation and analysis.
numpy: Library for numerical operations.
matplotlib.pyplot: Library for plotting graphs.
sklearn.preprocessing.MinMaxScaler: For scaling data to a specific range.
tensorflow.keras.models.Sequential: For building a sequential neural network.
tensorflow.keras.layers.LSTM, Dense, Dropout: Layers used in the neural network.
Fetching Historical Stock Data
python
Copy code
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
ticker: Stock symbol for Apple.
data: DataFrame containing historical stock data for the specified period.
Keeping Only the 'Close' Price Column
python
Copy code
data = data[['Close']]
data.dropna(inplace=True)
data[['Close']]: Selects only the 'Close' price column.
data.dropna(inplace=True): Removes rows with missing values.
Normalizing the Data
python
Copy code
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=['Close'], index=data.index)
MinMaxScaler: Scales data to the range [0, 1].
scaler.fit_transform(data): Fits the scaler to the data and transforms it.
pd.DataFrame(data_scaled, columns=['Close'], index=data.index): Converts the scaled data back to a DataFrame with the same index.
Plotting the Stock Price
python
Copy code
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
plt.figure(figsize=(10, 6)): Sets the figure size.
plt.plot(data.index, data['Close'], label='Close Price'): Plots the 'Close' price.
plt.title(f'{ticker} Stock Price'): Sets the title of the plot.
plt.xlabel('Date'): Labels the x-axis.
plt.ylabel('Price'): Labels the y-axis.
plt.legend(): Displays the legend.
plt.show(): Displays the plot.
Function to Create Sequences
python
Copy code
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences
create_sequences(data, seq_length): Function to create sequences of data for training the LSTM.
sequences: List to store sequences.
for i in range(len(data) - seq_length): Iterates over the data to create sequences.
seq = data[i:i + seq_length]: Creates a sequence of seq_length.
label = data[i + seq_length]: The label for the sequence.
sequences.append((seq, label)): Appends the sequence and label to the list.
Defining Sequence Length and Creating Sequences
python
Copy code
seq_length = 60
sequences = create_sequences(data_scaled.values, seq_length)
seq_length = 60: Defines the sequence length.
sequences = create_sequences(data_scaled.values, seq_length): Creates sequences from the scaled data.
Splitting the Data into Training and Testing Sets
python
Copy code
train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]
train_size = int(len(sequences) * 0.8): Calculates the size of the training set (80% of the data).
train_sequences = sequences[:train_size]: Training sequences.
test_sequences = sequences[train_size:]: Testing sequences.
Converting Sequences to NumPy Arrays
python
Copy code
X_train, y_train = zip(*train_sequences)
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = zip(*test_sequences)
X_test, y_test = np.array(X_test), np.array(y_test)
zip(*train_sequences): Unzips the training sequences into input and output.
np.array(X_train), np.array(y_train): Converts the training sequences to NumPy arrays.
zip(*test_sequences): Unzips the testing sequences into input and output.
np.array(X_test), np.array(y_test): Converts the testing sequences to NumPy arrays.
Building the LSTM Model
python
Copy code
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
Sequential(): Initializes the neural network model.
LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)): Adds an LSTM layer with 50 units, returning sequences, and input shape (seq_length, 1).
Dropout(0.2): Adds a dropout layer with a dropout rate of 20%.
LSTM(units=50, return_sequences=False): Adds another LSTM layer with 50 units, not returning sequences.
Dense(units=1): Adds a dense layer with 1 unit.
Compiling the Model
python
Copy code
model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mean_squared_error'): Compiles the model with Adam optimizer and mean squared error loss function.
Training the Model
python
Copy code
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1): Trains the model with 50 epochs, batch size of 32, and a validation split of 10%.
Making Predictions on the Test Data
python
Copy code
y_pred = model.predict(X_test)
y_pred = model.predict(X_test): Makes predictions on the test data.
Inverse Transforming the Predictions and Actual Values
python
Copy code
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)): Inverse transforms the actual values to their original scale.
y_pred_inv = scaler.inverse_transform(y_pred): Inverse transforms the predicted values to their original scale.
Plotting the Actual vs. Predicted Stock Prices
python
Copy code
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test_inv, label='Actual Price')
plt.plot(data.index[-len(y_test):], y_pred_inv, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
plt.figure(figsize=(10, 6)): Sets the figure size.
plt.plot(data.index[-len(y_test):], y_test_inv, label='Actual Price'): Plots the actual prices.
plt.plot(data.index[-len(y_test):], y_pred_inv, label='Predicted Price'): Plots the predicted prices.
plt.title(f'{ticker} Stock Price Prediction'): Sets the title of the plot.
plt.xlabel('Date'): Labels the x-axis.
plt.ylabel('Price'): Labels the y-axis.
plt.legend(): Displays the legend.
plt.show(): Displays the plot.
