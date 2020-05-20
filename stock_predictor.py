import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = 30, 20
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from stocks import get_data, get_scaler, train_lstm, prep_past_data, lstm_predict, plot_predicted_prices





if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters

    parser = argparse.ArgumentParser(description="swift 7-day stock prices predictor")

    parser.add_argument("--ticker", required=True, type=str, help="ticker name of the stocks that you have downloaded from Yahoo!")
    parser.add_argument("--root_dir", type=str, help="usually, type in Desktop", default=os.getcwd())
    parser.add_argument("--next_n_days", type=int, help="how many future days you want to see the price predictions", default=7)
    parser.add_argument("--epochs", type=int, help="how many training rounds you want the model to be trained", default=20)
    parser.add_argument('--output', action='store_true', 
    help="shows output")
   
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    # check if data file exists
    if not '{}/data/{}.csv'.format(args.root_dir, args.ticker):
        print('ticker data not found. Please upload a csv file!')
    
 
    # Read in csv training file
    ticker_string = args.ticker # 'PK' or 'PENN'
    
    # Read in EPOCHS
    EPOCHS = args.epochs
    # read in next_n_days if supplied by user
    if not args.next_n_days:
        next_n_days = 7
    else:
        next_n_days = args.next_n_days
        
    scaler_obj = get_scaler()
    
    # Train a LSTM model using the passed in data
    print('Model is off to a good training ...')
    print('\n')
    model = train_lstm(ticker_string, scaler_obj, timestep=10, epochs=EPOCHS)
    
    # prepare input data for prediction
    x_pred = prep_past_data(ticker_string, scaler_obj, 10)
    closing_prices = lstm_predict(x_pred, model, scaler_obj, next_n_days=next_n_days)
    plot_predicted_prices(ticker_string, closing_prices)
    
    print('\n')
    print('\n')
    print('The predicted prices for the next {} days are: \n'.format(next_n_days))
    for i in range(len(closing_prices)):
        print('Day {}'.format(i), '---$', float(closing_prices[i]))
        print('\n')
    print('\n')
    print('Pick the lowest day to BUY!')