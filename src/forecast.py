# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:04:39 2018

@author: Vinay Raman, PhD
"""
from pyspark.sql import window
from pyspark.sql.functions import col, asc, desc, to_timestamp,\
                                  unix_timestamp, from_unixtime
from pyspark.sql.types import StructType, StructField, LongType
import pyspark.sql.functions as func
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf
from pyspark.ml.regression import LinearRegression, \
                                  DecisionTreeRegressor,\
                                  RandomForestRegressor,\
                                  GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from TimeSeriesFeatureGenerator import LagGather, TrendGather, \
                                       MovingAverageSmoothing
                


def Forecast(df, forecast_days, nLags, \
             timeSeriesColumn, regressor, sparksession):
    
    # this performs model training
    # this calls the machine-learning algorithms of Spark ML library
    
    #labels for machine-learning
    LeadWindow = window.Window.rowsBetween(0,forecast_days)   
    df = df.withColumn("label",func.last(df[timeSeriesColumn]).over(LeadWindow))
    
    features = [timeSeriesColumn]
    
    #Auto-regression feature
    LagTransformer = LagGather()\
                     .setLagLength(nLags)\
                     .setInputCol(timeSeriesColumn)
    df = LagTransformer.transform(df)
    featuresGenerated = LagTransformer.getFeatureNames()
    features.extend(featuresGenerated)
    
    #Other feature generators here:
    #Moving Average Smoothing
    #TrendGather

#******************************************************************************
# VECTOR ASSEMBLER
    # this assembles the all the features 
    df = df.dropna()
    vA = VectorAssembler().setInputCols(features)\
                          .setOutputCol("features")
    df_m = vA.transform(df)
#******************************************************************************
# Splitting data into train, test
    splitratio = 0.7
    df_train, df_test = TimeSeriesSplit(df_m, splitratio, sparksession)
#******************************************************************************
# DECISION-TREE REGRESSOR
    if(regressor == "DecisionTreeRegression"):
           
        dr = DecisionTreeRegressor(featuresCol = "features",\
                                   labelCol = "label", maxDepth = 5)
        model = dr.fit(df_train)
        predictions_dr_test = model.transform(df_test)
        predictions_dr_train = model.transform(df_train)
        
        # RMSE is used as evaluation metric
        evaluator = RegressionEvaluator(predictionCol="prediction",\
                                        labelCol="label",\
                                        metricName ="r2")
        
        RMSE_dr_test = evaluator.evaluate(predictions_dr_test)
        RMSE_dr_train = evaluator.evaluate(predictions_dr_train)
        return (df_test, df_train, \
                predictions_dr_test, predictions_dr_train,\
                RMSE_dr_test, RMSE_dr_train)
#******************************************************************************
# LINEAR REGRESSOR
    if(regressor == 'LinearRegression'):
        lr = LinearRegression(featuresCol = "features", labelCol="label", \
                              maxIter = 100, regParam = 0.4, \
                              elasticNetParam = 0.1)
        model = lr.fit(df_train)
        predictions_lr_test = model.transform(df_test)
        predictions_lr_train = model.transform(df_train)
        
        # RMSE is used as evaluation metric
        evaluator = RegressionEvaluator(predictionCol="prediction",\
                                        labelCol="label",\
                                        metricName ="r2") 
        RMSE_lr_test= evaluator.evaluate(predictions_lr_test)
        RMSE_lr_train = evaluator.evaluate(predictions_lr_train)
        return (df_test, df_train, \
                predictions_lr_test, predictions_lr_train,\
                RMSE_lr_test, RMSE_lr_train)
    

#*****************************************************************************
# RANDOM FOREST REGRESSOR
    if(regressor == 'RandomForestRegression'):
        rfr = RandomForestRegressor(featuresCol="features",\
                                    labelCol="label",\
                                    maxDepth = 5,\
                                    subsamplingRate = 0.8,\
                                    )
        model = rfr.fit(df_train)
        predictions_rfr_test = model.transform(df_test)
        predictions_rfr_train = model.transform(df_train)
        
        # RMSE is used as evaluation metric
        evaluator = RegressionEvaluator(predictionCol="prediction",\
                                        labelCol="label",\
                                        metricName ="rmse")
        RMSE_rfr_test= evaluator.evaluate(predictions_rfr_test)
        RMSE_rfr_train = evaluator.evaluate(predictions_rfr_train)
        return (df_test, df_train, \
                predictions_rfr_test, predictions_rfr_train,\
                RMSE_rfr_test, RMSE_rfr_train)
    

#*****************************************************************************
# GRADIENT BOOSTING TREE REGRESSOR
    if(regressor == 'GBTRegression'):
        gbt = GBTRegressor(featuresCol="features",\
                           labelCol="label",\
                           maxDepth=5,\
                           subsamplingRate=0.8)
        
        model = gbt.fit(df_train)
        predictions_gbt_test = model.transform(df_test)
        predictions_gbt_train = model.transform(df_train)
        
        # RMSE is used as evaluation metric
        evaluator = RegressionEvaluator(predictionCol="prediction",\
                                        labelCol="label",\
                                        metricName ="rmse")
        
        RMSE_gbt_test= evaluator.evaluate(predictions_gbt_test)
        RMSE_gbt_train = evaluator.evaluate(predictions_gbt_train)
        return (df_test, df_train, \
                predictions_gbt_test, predictions_gbt_train,\
                RMSE_gbt_test, RMSE_gbt_train)
    

#*****************************************************************************
def Difference(df, inputCol, outputCol):
    # performs first-order differencing
    lag1Window = window.Window.rowsBetween(-1, 0)
    df = df.withColumn(outputCol, \
                       df[inputCol] - func.first(df[inputCol]).over(lag1Window))
    return df
#*****************************************************************************
def TimeSeriesSplit(df_m, splitRatio, sparksession):
    
    # Splitting data into train and test
    # we maintain the time-order while splitting
    # if split ratio = 0.7 then first 70% of data is train data
    # and remaining 30% of data is test data
    newSchema  = StructType(df_m.schema.fields + \
                [StructField("Row Number", LongType(), False)])
    new_rdd = df_m.rdd.zipWithIndex().map(lambda x: list(x[0]) + [x[1]])
    df_m2 = sparksession.createDataFrame(new_rdd, newSchema)
    total_rows = df_m2.count()
    splitFraction  =int(total_rows*splitRatio)
    df_train = df_m2.where(df_m2["Row Number"] >= 0)\
                   .where(df_m2["Row Number"] <= splitFraction)
    df_test = df_m2.where(df_m2["Row Number"] > splitFraction)
    
    return df_train, df_test
#*****************************************************************************
def CheckStationarity(timeSeriesCol):
    # this function works with Pandas dataframe only not with spark dataframes
    # this performs Augmented Dickey-Fuller's test
    
    test_result = adfuller(timeSeriesCol.values)
    print('ADF Statistic : %f \n' %test_result[0])
    print('p-value : %f \n' %test_result[1])
    print('Critical values are: \n')
    print(test_result[4])
        
#*****************************************************************************
def Predict(i, df1, df2, timeSeriesCol, predictionCol, joinCol):
    
    # this converts differenced predictions to raw predictions
    dZCol = 'DeltaZ'+str(i) 
    f_strCol = 'forecast_'+str(i)+'day'
    df = df1.join(df2, [joinCol], how="inner")\
                            .orderBy(asc("Date"))
    df = df.withColumnRenamed(predictionCol, dZCol)
    df = df.withColumn(f_strCol, col(dZCol)+col(timeSeriesCol))
    return df
#*****************************************************************************
def SavePredictions(df, \
                    timeSeriesCol,\
                    regressionType,\
                    forecast_days, \
                    feature_nLags,\
                    filename, \
                    sparksession):
    
    # this is the main function which calls forecast and predict
    # this saves predictions in csv files
    
    #Differencing data to remove non-stationarity
    diff_timeSeriesCol = "Diff_"+timeSeriesCol
    df = Difference(df, timeSeriesCol,diff_timeSeriesCol)
    
    RMSE_test = {}
    RMSE_train = {}
    
    #Forecasting and Undifferencing the data
    for i in range(1, forecast_days+1):
        
        # training with Spark's ML algorithms
        df_test, df_train, \
        predictions_test, predictions_train,\
        RMSE_ts, RMSE_tr = \
        Forecast(df.select("Date",timeSeriesCol,diff_timeSeriesCol),\
                 i, feature_nLags, \
                 diff_timeSeriesCol,regressionType, sparksession)

        RMSE_test.update({'forecast_'+str(i)+'day':RMSE_ts})
        RMSE_train.update({'forecast_'+str(i)+'day':RMSE_tr})
        
        #predictions for training data            
        if(i == 1):
            
            #saving the 1-day forecast as separate column
            corr_predict_train = Predict(i, \
                                         df_train.select("Row Number",\
                                                         "Date",\
                                                         timeSeriesCol),\
                                         predictions_train.select("Row Number",\
                                                                  "prediction"),
                                         timeSeriesCol,\
                                         "prediction",\
                                         "Row Number")
            
            corr_predict_test = Predict(i, \
                                        df_test.select("Row Number",\
                                                       "Date",\
                                                       timeSeriesCol),\
                                        predictions_test.select("Row Number",\
                                                                "prediction"),
                                        timeSeriesCol,\
                                        "prediction",\
                                        "Row Number") 
        else:
            
            # saving each subsequent forecast as separate column
            strCol_prev= "forecast_" + str(i-1) + "day"

            corr_predict_train = Predict(i, \
                                         corr_predict_train,\
                                         predictions_train.select("Row Number",\
                                                                 "prediction"),\
                                         strCol_prev,\
                                         "prediction",\
                                         "Row Number")
            corr_predict_test = Predict(i, \
                                        corr_predict_test,\
                                        predictions_test.select("Row Number",\
                                                                "prediction"),\
                                        strCol_prev,\
                                        "prediction",\
                                        "Row Number")

        # saving actual labels as separate columns
        LeadWindow = window.Window.rowsBetween(0, i)    
        a_strCol = "actual_"+str(i)+"day"
        corr_predict_test = corr_predict_test.withColumn(\
                            a_strCol, \
                            func.last(corr_predict_test[timeSeriesCol])\
                                .over(LeadWindow))
        corr_predict_train = corr_predict_train.withColumn(\
                             a_strCol, \
                             func.last(corr_predict_test[timeSeriesCol])\
                                 .over(LeadWindow))

    # Saving data into csv files
    corr_predict_test.write.format("csv").option("header","true")\
                     .save(filename+"test.csv")
    corr_predict_train.write.format("csv").option("header","true")\
                     .save(filename+"train.csv") 
    
    #error statistics summary  
    print("Error statistics summary for %s " %(filename))
    print("RMSE for train data:\n")
    print(RMSE_train)
    print("RMSE for test data:\n")
    print(RMSE_test)
    print('Two output files created')
    print('Predictions for train data: %s' %(filename+'train.csv'))
    print('Predictions for test data: %s' %(filename +'test.csv'))
    return RMSE_train, RMSE_test
#*****************************************************************************

        
