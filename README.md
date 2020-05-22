# Stock Prices Predictor using Command Line Interface
**(More features to come such as directly 10-k document words analysis, different statistical testing, etc.)**
## Introduction

This repository contains an easy-to-use command-line stock predictor that can make predictions on a particular stock's prices on nultiple days in the future. The underlying model is a 2-layer LSTM neural network. It is simple yet very effective in comparison to other more traditional time series forcasting models.

Once you have fired up your terminal, the only thing you need to type in is the ticker name of the stock of your interest. Optionally, you can type in how many days' predictions you want to see in the future. Finally, you will see a list of price predictions of this stock. The goal is to quickly help you make buy&sell decisions. (Note, I will not be responsible to any monetary loss if you happen to use this application!).

## Command Line Prediction Output Should Look Like ...

!<img width="649" alt="commandline" src="https://user-images.githubusercontent.com/43501958/82501157-a6fd7380-9aa9-11ea-8a17-07f11cdf476d.JPG">

## Python Libraries Used for this project

* pandas, matplotlib, numpy
* pylab
* pyramid -- for ARIMA forecasting
* sklearn, keras
* More details can be found in the script

## Brief walkthrough of the files in this repo

* stocks.py is where all the functions are. This includes data reads, data preprocessing, model training, model inference, and plotting functions.
* stock_predictor.py is the script that you will run through the terminal.
* **stock_predictor_dev.ipynb** is where I prototyped this command line application.
   * It includes all the model selection process. 
   * You can see the path I took to finally decide upon using this LSTM model as the final implementation.
   * You can see different metrics I used for model selection. 
   * You can also play with different loss functions and hyperparameters to retrain the model
   
 * data folder includes different stock prices downloaded from Yahoo finance.
 
 ## How to use the predictor?
 
 * Download all files to one folder and 'cd' into this folder from the terminal
 * `python stock_predictor.py --ticker=PK --epochs=20` 
 * Yes, you can not only choose what ticker to predict the prices for but also how many epochs you want to train
 * You can also choose how many days you want to predict the prices for in the future.
    `python stock_predictor.py --ticker=PK --epochs=20 --next_n_days=20`
    
 ## References
 
 See a list of references in the project documentation
