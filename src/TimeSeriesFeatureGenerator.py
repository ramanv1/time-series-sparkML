# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 08:29:47 2018

@author: Vinay Raman, PhD
"""

import numpy as np
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, lead, lit, avg, signum
import pyspark.sql.functions as func

class LagGather:
    
    # generates features for machine-learning
    # previous time-step values are used as features
    
    def __init__(self):
        # this class has 2 data members
        self.nLags = 0
        self.FeatureNames = []
        
    def setLagLength(self, nLags):
        # this method sets the lag-length
        # if we want only previous time-step as feature 
        # use lag-length = 1
        # if we want more lagged time-steps as features
        # set higher lag-length
        self.nLags = nLags
        return self
    
    def setInputCol(self, colname):
        #sets the input col for which features are generated
        # this identifies the univariate time-series on 
        # which machine-learning and forecasting is done
        self.columnName = colname
        return self
    
    def transform(self, df):
        # transforms the spark dataframe and creates columns
        # that have time-lagged values
        # columns generated as used as features in ML
        df = df.withColumn("Series",lit('Univariate'))       
        mywindow = Window.orderBy("Series")
        for i in range(self.nLags):
            strLag = self.columnName+'_LagBy_'+str(i+1)
            df = df.withColumn(strLag,lag(\
                                df[self.columnName], i+1).over(mywindow))
            self.FeatureNames.append(strLag) 
            
        df = df.drop("Series")
        return df
    
    def getFeatureNames(self):
        # this return the names of feature-columns that are
        # generated by transform method
        return self.FeatureNames

class MovingAverageSmoothing:
    # this class is used for performing Moving-average smoothing

    def __init__(self):
        # this class has 2 data members
        self.nLags= 0
        self.FeatureNames = []
        
    def setLagLength(self, nLags):
        # this sets the window size over which moving average is performed
        self.nLags = nLags
        return self
    
    def setInputCol(self, colname):
        #this sets the time-series column on which 
        #moving-average is performed
        self.columnName = colname
        return self
    
    def transform(self, df):
        # this transforms the spark dataframe (i.e time-series column)
        # and creates column contain the moving-average over created 
        # time-window
        mywindow = Window.rowsBetween(-self.nLags, 0)
        strMovAvg = self.columnName+'_'\
                    + str(self.nLags)+'_MovingAvg'
        df = df.withColumn(strMovAvg,\
                           avg(df[self.columnName]).over(mywindow))
        self.FeatureNames.append(strMovAvg)
        return df 
    
    def getFeatureNames(self):
        # this returns the name of feature-column 
        # created by transform method    
        return self.FeatureNames

class TrendGather:
    
    # this class is used to find trend in time-series data
    def __init__(self):
        # this has 2 data members
        self.nLags= 0 
        self.FeatureNames= []
        
    def setLagLength(self, nLags):
        # this sets the window-size over which trend is determined
        self.nLags = nLags
        return self
    
    def setInputCol(self, colname):
        # this sets the time-series column for which trend is to be
        # determined
        self.columnName = colname
        return self
    
    def transform(self, df):
        # this transforms the spark-dataframe (i.e. time-series column)
        # and generates column containing values +1, or -1 
        # if current value > time-lagged value then column-value = +1.0
        # if current value < time-lagged value then column-value = -1.0
        df = df.withColumn("Series",lit('Univariate'))       
        mywindow = Window.orderBy("Series")
        for i in range(self.nLags):
            strSign = self.columnName +'_Lag_'+str(i+1)+'_Sign'
            df = df.withColumn(strSign,\
                               signum((df[self.columnName] - \
                                       lag(df[self.columnName],i+1)\
                                       .over(mywindow))))
            self.FeatureNames.append(strSign)
        df = df.drop("Series")
        return df
    
    def getFeatureNames(self):
        # this returns name of feature generted by transform method
        return self.FeatureNames


        
