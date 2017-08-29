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
    trainSize=int(len(rawArrayDatas[0])*0.7)
    testSize=len(rawArrayDatas[0])-trainSize

    testY= rawArrayDatas[1][trainSize:]

####################################################################################LSTM

    testY= rawArrayDatas[1][trainSize:]
    mockDs = rawArrayDatas[0][:trainSize]
    mockY = list((rawArrayDatas[1][:trainSize]))
    mockSales = list(zip(mockDs, mockY))
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "traindata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ds = rawArrayDatas[0]
    y = list(rawArrayDatas[1])
    sales = list(zip(ds, y))

    mockForecastDictionary['LSTM']= LSTM(mockDs, mockY, testSize)
    realForecastDictionary['LSTM'] = LSTM(ds,y, forecastDay)




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


def LSTM(sales, forecastDay):
    txs = pd.DataFrame(data=sales, columns=['date', 'sales'])
    year = lambda x: datetime.strptime(x, "%Y-%m-%d").year
    day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d").weekday()
    month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
    # please read docs on how week numbers are calculate
    week_number = lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%V')

    txs['year'] = txs['date'].map(year)
    txs['month'] = txs['date'].map(month)
    txs['week_number'] = txs['date'].map(week_number)
    txs['day_of_week'] = txs['date'].map(day_of_week)

    seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]  # dec - feb is winter, then spring, summer, fall etc
    season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d").month - 1)]
    day_of_week01s = [0, 0, 0, 0, 0, 1, 1]
    day_of_week01 = lambda x: day_of_week01s[(datetime.strptime(x, "%Y-%m-%d").weekday())]
    txs['season'] = txs['date'].map(season)
    txs['day_of_week01'] = txs['date'].map(day_of_week01)
    originalSales = list(txs['sales'])
    sales = list(txs['sales'])

    tempxy = [list(txs['season']), list(txs['day_of_week']), list(txs['week_number']), sales]

    xy = np.array(tempxy).transpose().astype(np.float)
    originalXY = np.array(tempxy).transpose().astype(np.float)
    xy = minMaxNormalizer(xy)
    # data_dim은 y값 도출을 위한 feature 가지수+1(독립변수 가지수 +1(y포함))
    data_dim = 4

    # data_dim크기의 data 한 묶음이 seq_length만큼 input으로 들어가
    seq_length = 5

    # output_dim(=forecastDays)만큼의 다음날 y_data를 예측
    forecastDays = 7
    output_dim = forecastDays

    # hidden_dim은 정말 임의로 설정
    hidden_dim = 10

    # learning rate은 배우는 속도(너무 크지도, 작지도 않게 설정)
    learning_rate = 0.01

    # iterations는 반복 횟수
    iterations = 1000

    x = xy
    y = xy[:, [-1]]

    # build a series dataset(seq_length에 해당하는 전날 X와 다음 forecastDays에 해당하는 Y)
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length - forecastDays):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length:i + seq_length + forecastDays]
        _y = np.reshape(_y, (forecastDays))
        #     _y=Y[i+seq_length:i+seq_length+forecastDays]
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, forecastDays])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.nn.relu)

    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss)

    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
    with tf.Session() as sess:
        # 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            # print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = minMaxDeNormalizer(sess.run(Y_pred, feed_dict={X: testX}), originalXY)

    return list(test_predict[-1])


def Bayseian(rawArrayDatas, processId, forecastDay):
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