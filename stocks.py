import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = 30, 20
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# helper functions
def get_data(symbol_list):
    
    for sym in symbol_list:
        df_adj_closed = pd.read_csv('data/{}.csv'.format(sym), index_col='Date', parse_dates=True, na_values=['nan'], usecols=['Date', 'Adj Close'])
        df_adj_closed = df_adj_closed.rename(columns={'Adj Close': 'adj_close'})
        df_adj_closed.dropna(inplace=True)
    
    return df_adj_closed


def get_scaler():
    
    return MinMaxScaler(feature_range=(0, 1))

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# multiple-step prediction input preparation
def split_sequence_mult(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def reshape_data(price_X):
    
    # since this is a univariate time series, feature = 1, by default
    return price_X.reshape((price_X.shape[0], price_X.shape[1], 1))

def config_model(timestep, num_features=1, n_next_days=1):
    
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(timestep, num_features)))
    model.add(LSTM(units=16))
    model.add(Dense(n_next_days))

    model.compile(loss='mse', optimizer='adam')
    
    return model
    

def train_lstm(tick_name, scaler_obj, timestep=10, epochs=20):
    
    # get data series
    df_tick = get_data([tick_name])
    #dataset_tick = df_tick.values
    ruth_ary = np.array(ruth.adj_close)
    ruth_X, ruth_y = split_sequence(ruth_ary, 20)
    
    # split into train and test
    num_train = int(len(dataset_tick) * 0.8)
    train_tick = dataset_tick[0:num_train, :]
    valid_tick = dataset_tick[num_train:, :]
    
    # scale the data
    scaled_tick_data = scaler_obj.fit_transform(dataset_tick)
    
    # make training data for LSTM model
    x_tick_train, y_tick_train = [], []
    for i in range(timestep, len(train_tick)):
        x_tick_train.append(scaled_tick_data[i-timestep:i,0])
        y_tick_train.append(scaled_tick_data[i,0])
        
    x_tick_train, y_tick_train = np.array(x_tick_train), np.array(y_tick_train)

    x_tick_train = np.reshape(x_tick_train, (x_tick_train.shape[0], x_tick_train.shape[1], 1))
    
    # create and fit the LSTM network
    model_tick = Sequential()
    model_tick.add(LSTM(units=32, return_sequences=True, input_shape=(x_tick_train.shape[1],1)))
    model_tick.add(LSTM(units=16))
    model_tick.add(Dense(1))

    model_tick.compile(loss='mean_squared_error', optimizer='adam')
    model_tick.fit(x_tick_train, y_tick_train, batch_size=1, epochs=epochs, verbose=2)
    
    return model_tick


def prep_past_data(tick_name, scaler_obj, timestep=10):
    
    #get data series
    df_tick = get_data([tick_name])
    dataset_tick = df_tick.values
    
    num_valid = len(dataset_tick)-int(len(dataset_tick) * 0.8)
    inputs = df_tick[len(dataset_tick) - num_valid - timestep:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler_obj.transform(inputs)

    X_pred_tick = []
    for i in range(timestep,inputs.shape[0]):
        X_pred_tick.append(inputs[i-timestep:i,0])
        
    X_pred_tick = np.array(X_pred_tick)

    X_pred_tick = np.reshape(X_pred_tick, (X_pred_tick.shape[0], X_pred_tick.shape[1],1))
    
    return X_pred_tick

def lstm_predict(X_pred_tick, model_tick, scaler_obj, next_n_days):
    # make prediction
    closing_prices = model_tick.predict(X_pred_tick)
    closing_prices = scaler_obj.inverse_transform(closing_prices)
    
    return closing_prices[:next_n_days]


#for plotting
def plot_predicted_prices(tick_name, closing_tick_prices):

    df_tick = get_data([tick_name])
    train_tick = df_tick[:int(df_tick.shape[0] * 0.8)]
    valid_tick = df_tick[int(df_tick.shape[0] * 0.8):][:len(closing_tick_prices)]

    
    ax = plt.gca()

    valid_tick['predictions'] = closing_tick_prices
    plt.plot(train_tick['adj_close'], color = 'yellow')
    plt.plot(valid_tick['adj_close'], color = 'brown')
    plt.plot(valid_tick['predictions'], color = 'green', linewidth=4)
    plt.show()
    
## usage

## use the past 191 days' prices to predict the following 7 days
# scaler_obj = get_scaler()
# model_penn = train_lstm('PENN', scaler_obj, timestep=10)
# x_pred_penn = prep_past_data('PENN', scaler_obj, 10)
# closing_penn_prices = lstm_predict(x_pred_penn, model_penn, scaler_obj, next_n_days=7)
# plot_predicted_prices('PENN', closing_penn_prices)

