# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:33:25 2018

@author: Owner
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:27:39 2018

@author: Owner
"""

from pyspark.sql import window
from pyspark.sql.functions import col, asc, desc, to_timestamp,\
                                  unix_timestamp, from_unixtime
from pyspark.sql.types import StructType, StructField, LongType
import pyspark.sql.functions as func
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import quandl
from forecast import Forecast, Difference, Predict, SavePredictions


if __name__=="__main__":

    #initializing local variables    
    RMSE_train_df = pd.DataFrame()
    RMSE_test_df = pd.DataFrame()
    
    #setting up spark environment and create spark session object
    conf = SparkConf()
    spark = SparkSession.builder.appName("TimeSeries").master("local")\
                            .config(conf=conf).getOrCreate()
    
    #reading saved data from local folder
    apple_data = spark.read.format("csv").option("inferSchema","true")\
                      .option("header","true")\
                      .load("./apple.csv")
    fb_data =  spark.read.format("csv").option("inferSchema","true")\
                      .option("header","true")\
                      .load("./fb.csv")
    google_data = spark.read.format("csv").option("inferSchema","true")\
                      .option("header","true")\
                      .load("./google.csv")
    netflix_data = spark.read.format("csv").option("inferSchema","true")\
                      .option("header","true")\
                      .load("./netflix.csv")
    
    #Performing 1, 2, 3, 4, and 5-day forecasts 
    #Features used:
    #Values at time-step t-3, t-2, t-1, t
    
    # setting up parameters for simulation
    timeSeriesCol = "Close"
    regressionType = "RandomForestRegression"
    forecast_days = 5
    num_lags = 3
    
    # Machine-learning and forecasting for Apple stock
    RMSE_train, RMSE_test = SavePredictions(apple_data, \
                                            timeSeriesCol,\
                                            regressionType,\
                                            forecast_days,\
                                            num_lags,\
                                            "RFR_ApplePredictions",\
                                            spark)
    RMSE_train_df = RMSE_train_df.append(RMSE_train, ignore_index = True) 
    RMSE_test_df = RMSE_test_df.append(RMSE_test, ignore_index = True)
    
    #Machine-learning and forecasting for Facebook stock
    RMSE_train, RMSE_test = SavePredictions(fb_data, \
                                            timeSeriesCol,\
                                            regressionType,\
                                            forecast_days,\
                                            num_lags,\
                                            "RFR_FacebookPredictions",\
                                            spark)
    RMSE_train_df = RMSE_train_df.append(RMSE_train, ignore_index = True) 
    RMSE_test_df = RMSE_test_df.append(RMSE_test, ignore_index = True)
    
    #Machine-learning and forecasting for Google stock
    RMSE_train, RMSE_test = SavePredictions(google_data,\
                                            timeSeriesCol,\
                                            regressionType,\
                                            forecast_days,\
                                            num_lags,\
                                            "RFR_GooglePredictions",\
                                            spark)
    RMSE_train_df = RMSE_train_df.append(RMSE_train, ignore_index = True) 
    RMSE_test_df = RMSE_test_df.append(RMSE_test, ignore_index = True)
    
    #Machine-learning and forecasting for Netflix stock
    RMSE_train, RMSE_test =  SavePredictions(netflix_data,\
                                             timeSeriesCol,\
                                             regressionType,\
                                             forecast_days,\
                                             num_lags,\
                                             "RFR_NetflixPredictions",\
                                             spark)
    RMSE_train_df = RMSE_train_df.append(RMSE_train, ignore_index = True) 
    RMSE_test_df = RMSE_test_df.append(RMSE_test, ignore_index = True)
    
    # Saving RMSE statistics
    RMSE_test_df['Ticker']= ['AAPL','FB','GOOGL','NFLX']
    RMSE_train_df['Ticker']=['AAPL','FB','GOOGL','NFLX']
    RMSE_test_df = RMSE_test_df.set_index('Ticker')
    RMSE_train_df = RMSE_train_df.set_index('Ticker')
    
    fn = regressionType +'.csv'
    RMSE_test_df.to_csv("RMSE_test_"+fn)
    RMSE_train_df.to_csv("RMSE_train_"+fn)