from Settings import DefineManager
from Utils import LoggingManager
from . import FirebaseDatabaseManager
import pandas as pd
import numpy as np
from fbprophet import Prophet

def LearningModuleRunner(rawArrayDatas, processId, forecastDay):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    mockForecastDictionary={}
    realForecastDictionary={}

    trainSize=int(len(rawArrayDatas[0])*0.7)
    testSize=len(rawArrayDatas[0])-trainSize

    testY= rawArrayDatas[1][trainSize:]
####################################################################################LSTM
    mockMinData = np.min(rawArrayDatas[1][:trainSize])
    mockMaxData = np.max(rawArrayDatas[1][:trainSize])
    mockForecastDictionary['LSTM'] = [i+mockMinData for i in np.random.beta(mockMinData, mockMaxData, testSize)*(mockMaxData-mockMinData)]

    minData = np.min(rawArrayDatas[1])
    maxData = np.max(rawArrayDatas[1])
    realForecastDictionary['LSTM'] = [i+minData for i in np.random.beta(minData, maxData, forecastDay)*(maxData-minData)]
####################################################################################BAYSEIAN

    mockDs = rawArrayDatas[0][:trainSize]
    mockY = list(np.log(rawArrayDatas[1][:trainSize]))
    mockSales = list(zip(mockDs, mockY))
    mockPreprocessedData = pd.DataFrame(data=mockSales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "traindata create success",
                                   DefineManager.LOG_LEVEL_INFO)

    ds=rawArrayDatas[0]
    y=list(np.log(rawArrayDatas[1]))
    sales=list(zip(ds, y))
    preprocessedData=pd.DataFrame(data=sales, columns=['ds','y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realdata create success",
                                   DefineManager.LOG_LEVEL_INFO)


    mockModel = Prophet()
    mockModel.fit(mockPreprocessedData)
    mockFuture = mockModel.make_future_dataframe(periods=testSize)
    mockForecastProphetTable= mockModel.predict(mockFuture)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "mockforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    mockForecastDictionary['Bayseian'] = [np.exp(y) for y in mockForecastProphetTable['yhat'][-testSize:]]

    model=Prophet()
    model.fit(preprocessedData)
    future = model.make_future_dataframe(periods=forecastDay)
    forecastProphetTable = model.predict(future)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "realforecast success",
                                   DefineManager.LOG_LEVEL_INFO)
    realForecastDictionary['Bayseian']=[np.exp(y) for y in forecastProphetTable['yhat'][-forecastDay:]]

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

    FirebaseDatabaseManager.StoreOutputData(processId, resultArrayData=data, resultArrayDate=date, status=DefineManager.ALGORITHM_STATUS_DONE)
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