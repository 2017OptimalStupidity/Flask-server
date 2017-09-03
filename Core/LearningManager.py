from Settings import DefineManager
from Utils import LoggingManager
from . import FirebaseDatabaseManager
import pandas as pd
import numpy as np
from fbprophet import Prophet
import tensorflow as tf
import matplotlib
import os
import matplotlib.pyplot as plt
from datetime import datetime
tf.set_random_seed(77)

mockForecastDictionary = {}
realForecastDictionary = {}

def LearningModuleRunner(rawArrayDatas, processId, forecastDay):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    global mockForecastDictionary
    global realForecastDictionary
    mockForcastDay=2*forecastDay





    ##Make txsForRealForecastLstm   [:]
    ds = rawArrayDatas[0]
    y = list(rawArrayDatas[1])
    sales = list(zip(ds, y))
    txsForRealForecastLstm =pd.DataFrame(data=sales, columns=['date', 'sales'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "txsForRealForecastLstm create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ##Make txsForRealForecastBayesian [:-forecastDay] & np.log
    ds = rawArrayDatas[0][:-forecastDay]
    #TODO bayseian에 대해서는 input값이 0인 상황처리 필요
    y= list(np.log(rawArrayDatas[1][:-forecastDay] ))
    sales = list(zip(ds, y))
    txsForRealForecastBayesian =pd.DataFrame(data=sales, columns=['date', 'sales'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "txsForRealForecastBayesian create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ##Make txsForMockForecastLstm [:-forecastDay]
    ds = rawArrayDatas[0][:-forecastDay]
    y= list(rawArrayDatas[1][:-forecastDay] )
    sales = list(zip(ds, y))
    txsForMockForecastLstm =pd.DataFrame(data=sales, columns=['date', 'sales'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "txsForMockForecastLstm create success",
                                   DefineManager.LOG_LEVEL_INFO)
    ##Make txsForMockForecastBayseian   [:-3*forecastDay] & np.log
    ds = rawArrayDatas[0][:-3*forecastDay]
    #TODO bayseian에 대해서는 input값이 0인 상황처리 필요
    y= list(np.log(rawArrayDatas[1][:-3*forecastDay]))
    sales = list(zip(ds, y))
    txsForMockForecastBayseian =pd.DataFrame(data=sales, columns=['date', 'sales'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "txsForMockForecastBayseian create success",
                                   DefineManager.LOG_LEVEL_INFO)

    #testY for algorithm compare has size of 2*forecastDay:  rawArrayDatas[1][-3*forecastDay:-forecastDay]
    testY= rawArrayDatas[1][-3*forecastDay:-forecastDay]


    dayOrWeekOrMonth='week' # 'day', 'week', 'month'
    if dayOrWeekOrMonth=='day':
        ####Bayseian



        ###########date 받아오기, mockForecast 에 저장

        ####LSTM
        mockForecastDictionary['LSTM'] = LSTM(sales, mockForcastDay)
        LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "mockForecast success",
                                       DefineManager.LOG_LEVEL_INFO)


        nameOfBestAlgorithm= AlgorithmCompare(testY)

        ####더 좋은 알고리즘 호출
        if nameOfBestAlgorithm=='LSTM':
            tf.reset_default_graph()
            realForecastDictionary['LSTM'] = LSTM(sales+forecastDay에 해당하는 공행렬, forecastDay)
            LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "LSTMrealForecast success",
                                           DefineManager.LOG_LEVEL_INFO)

        elif nameOfBestAlgorithm=='Bayseian':

        ###########결과: realForecast에 저장


        data = rawArrayDatas[1] + realForecastDictionary[nameOfBestAlgorithm]

        FirebaseDatabaseManager.StoreOutputData(processId, resultArrayData=data, resultArrayDate=date,
                                                status=DefineManager.ALGORITHM_STATUS_DONE)
    elif dayOrWeekOrMonth=='week':

####################################################################################LSTM


    # mockMinData = np.min(rawArrayDatas[1][:trainSize])
    # mockMaxData = np.max(rawArrayDatas[1][:trainSize])
    # mockForecastDictionary['LSTM'] = [i+mockMinData for i in np.random.beta(mockMinData, mockMaxData, testSize)*(mockMaxData-mockMinData)]
    #
    # minData = np.min(rawArrayDatas[1])
    # maxData = np.max(rawArrayDatas[1])
    # realForecastDictionary['LSTM'] = [i+minData for i in np.random.beta(minData, maxData, forecastDay)*(maxData-minData)]

####################################################################################BAYSEIAN
    trainSize = int(len(rawArrayDatas[0]) * 0.7)
    testSize = len(rawArrayDatas[0]) - trainSize

    testY = rawArrayDatas[1][trainSize:]
    mockDs = rawArrayDatas[0][:trainSize]
    mockY = list(np.log(rawArrayDatas[1][:trainSize]))
    mockSales = list(zip(mockDs, mockY))
    mockPreprocessedData = pd.DataFrame(data=mockSales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "traindata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ds = rawArrayDatas[0]
    y = list(np.log(rawArrayDatas[1]))
    sales = list(zip(ds, y))
    preprocessedData = pd.DataFrame(data=sales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realdata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    mockModel = Prophet(yearly_seasonality=True)
    mockModel.fit(mockPreprocessedData)
    mockFuture = mockModel.make_future_dataframe(periods=testSize)
    mockForecastProphetTable = mockModel.predict(mockFuture)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "mockforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    mockForecastDictionary['Bayseian'] = [np.exp(y) for y in mockForecastProphetTable['yhat'][-testSize:]]

    model = Prophet(yearly_seasonality=True)
    model.fit(preprocessedData)
    future = model.make_future_dataframe(periods=forecastDay)
    forecastProphetTable = model.predict(future)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    realForecastDictionary['Bayseian'] = [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

    date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
##################################################################################################ALGORITHM COMPARE
    nameOfBestAlgorithm = 'LSTM'
    minData = rmse(testY, mockForecastDictionary[nameOfBestAlgorithm])
    rms = 0
    for algorithm in realForecastDictionary.keys():
        rms = rmse(testY, mockForecastDictionary[algorithm])
        if rms < minData:
            nameOfBestAlgorithm = algorithm

    data = rawArrayDatas[1] + realForecastDictionary[nameOfBestAlgorithm]

    FirebaseDatabaseManager.StoreOutputData(processId, resultArrayData=data, resultArrayDate= date, status=DefineManager.ALGORITHM_STATUS_DONE)
    return

def LSTM(txs, forecastDay, features):

    #Add basic date related features to the table
    year = lambda x: datetime.strptime(x, "%Y-%m-%d").year
    dayOfWeek = lambda x: datetime.strptime(x, "%Y-%m-%d").weekday()
    month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
    weekNumber = lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%V')
    txs['year'] = txs['date'].map(year)
    txs['month'] = txs['date'].map(month)
    txs['weekNumber'] = txs['date'].map(weekNumber)
    txs['dayOfWeek'] = txs['date'].map(dayOfWeek)

    #Add non-basic date related features to the table
    seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]  # dec - feb is winter, then spring, summer, fall etc
    season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d").month - 1)]
    day_of_week01s = [0, 0, 0, 0, 0, 1, 1]
    day_of_week01 = lambda x: day_of_week01s[(datetime.strptime(x, "%Y-%m-%d").weekday())]
    txs['season'] = txs['date'].map(season)
    txs['dayOfWeek01'] = txs['date'].map(day_of_week01)

    #Backup originalSales
    originalSales = list(txs['sales'])
    sales = list(txs['sales'])

    if features =='DayOfWeek_WeekNumber_Month_Season' :
        tempxy = [list(txs['dayOfWeek']), list(txs['weekNumber']),list(txs['month']),list(txs['season']) , sales]
    elif features =='DayOfWeek01_WeekNumber_Month_Season' :
        tempxy = [list(txs['dayOfWeek01']), list(txs['weekNumber']),list(txs['month']),list(txs['season']) , sales]

    elif features =='WeekNumber_Month_Season_Year' :
        tempxy = [list(txs['weekNumber']), list(txs['month']), list(txs['season']),list(txs['year']), sales]

    xy = np.array(tempxy).transpose().astype(np.float)

    #Backup originalXY for denormalize
    originalXY = np.array(tempxy).transpose().astype(np.float)
    xy = minMaxNormalizer(xy)

    #TRAIN PARAMETERS
    # data_dim은 y값 도출을 위한 feature 가지수+1(독립변수 가지수 +1(y포함))
    data_dim = 5
    # data_dim크기의 data 한 묶음이 seq_length만큼 input으로 들어가
    seq_length = 5
    # output_dim(=forecastDays)만큼의 다음날 y_data를 예측
    output_dim = forecastDay
    # hidden_dim은 정말 임의로 설정
    hidden_dim = 10
    # learning rate은 배우는 속도(너무 크지도, 작지도 않게 설정)
    learning_rate = 0.001
    # iterations는 반복 횟수
    iterations = 500

    # Build a series dataset(seq_length에 해당하는 전날 X와 다음 forecastDays에 해당하는 Y)
    x = xy
    y = xy[:, [-1]]
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length - forecastDay):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length:i + seq_length + forecastDay]
        _y = np.reshape(_y, (forecastDay))

        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, forecastDay])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "train prepare success",
                                   DefineManager.LOG_LEVEL_INFO)
    with tf.Session() as sess:
        # 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            # print("[step: {}] loss: {}".format(i, step_loss))
        LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "test success",
                                       DefineManager.LOG_LEVEL_INFO)
        # Test step
        test_predict = (sess.run(Y_pred, feed_dict={X: testX}))
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "test success",
                                   DefineManager.LOG_LEVEL_INFO)
    return list(minMaxDeNormalizer(test_predict[-1], originalXY))


def Bayseian(txs, forecastDay, unit):
    global mockForecastDictionary
    global realForecastDictionary

    if unit is 'day':
        model = Prophet()
        model.fit(txs)
        future = model.make_future_dataframe(periods=forecastDay)
        forecastProphetTable = model.predict(future)
        LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
        realForecastDictionary['Bayseian'] = [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

        date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]
    elif unit is 'week':
        model = Prophet()
        model.fit(txs)
        future = model.make_future_dataframe(periods=forecastDay)
        forecastProphetTable = model.predict(future)
        LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realforecast success",
                                       DefineManager.LOG_LEVEL_INFO)
        realForecastDictionary['Bayseian'] = [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

        date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]

    return date

def BayseianWeek(rawArrayDatas, processId, forecastDay):
    global mockForecastDictionary
    global realForecastDictionary

    trainSize = int(len(rawArrayDatas[0]) * 0.7)
    testSize = len(rawArrayDatas[0]) - trainSize

    testY = rawArrayDatas[1][trainSize:]
    mockDs = rawArrayDatas[0][:trainSize]
    mockY = list(np.log(rawArrayDatas[1][:trainSize]))
    mockSales = list(zip(mockDs, mockY))
    mockPreprocessedData = pd.DataFrame(data=mockSales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "traindata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ds = rawArrayDatas[0]
    y = list(np.log(rawArrayDatas[1]))
    sales = list(zip(ds, y))
    preprocessedData = pd.DataFrame(data=sales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realdata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    mockModel = Prophet()
    mockModel.fit(mockPreprocessedData)
    mockFuture = mockModel.make_future_dataframe(periods=testSize)
    mockForecastProphetTable = mockModel.predict(mockFuture)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "mockforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    mockForecastDictionary['Bayseian'] = [np.exp(y) for y in mockForecastProphetTable['yhat'][-testSize:]]

    model = Prophet()
    model.fit(preprocessedData)
    future = model.make_future_dataframe(periods=forecastDay)
    forecastProphetTable = model.predict(future)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    realForecastDictionary['Bayseian'] = [np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

    date = [d.strftime('%Y-%m-%d') for d in forecastProphetTable['ds']]

    return date


def rmse(a,b):
    sum=0
    for i in range(len(a)):
        sum=sum+(a[i]-b[i])**2
    return np.sqrt(sum/len(a))

def minMaxNormalizer(data):
    numerator=data-np.min(data)
    denominator=np.max(data)-np.min(data)
    return numerator/(denominator+1e-7)

def minMaxDeNormalizer(data, originalData):
    shift=np.min(originalData)
    multiplier=np.max(originalData)-np.min(originalData)
    return (data+shift)*multiplier

def AlgorithmCompare(testY):
    global mockForecastDictionary
    global realForecastDictionary
    nameOfBestAlgorithm = 'LSTM'
    minData = rmse(testY, mockForecastDictionary[nameOfBestAlgorithm])
    rms = 0
    for algorithm in realForecastDictionary.keys():
        rms = rmse(testY, mockForecastDictionary[algorithm])
        if rms < minData:
            nameOfBestAlgorithm = algorithm

    return nameOfBestAlgorithm


def ProcessResultGetter(processId):

    status=FirebaseDatabaseManager.GetOutputDataStatus(processId)

    if(status==DefineManager.ALGORITHM_STATUS_DONE):
        date= FirebaseDatabaseManager.GetOutputDateArray(processId)
        data= FirebaseDatabaseManager.GetOutputDataArray(processId)
        return [date, data], DefineManager.ALGORITHM_STATUS_DONE
    elif(status==DefineManager.ALGORITHM_STATUS_WORKING):
        return [[], DefineManager.ALGORITHM_STATUS_WORKING]
    else:
        LoggingManager.PrintLogMessage("LearningManager", "ProcessResultGetter",
                                       "process not available #" + str(processId), DefineManager.LOG_LEVEL_ERROR)
        return [[], DefineManager.ALGORITHM_STATUS_WORKING]
