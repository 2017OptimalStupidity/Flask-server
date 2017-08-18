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
    for yhat, y in mockForecastDictionary['LSTM'], testY:
        sum = sum + (yhat - y) ** 2
    minRmse=sum

    for k,v in mockForecastDictionary.items():
        sum=0
        for yhat, y in v, testY:
            sum=sum+(yhat-y)**2
        if sum<minData:
            nameOfBestAlgorithm=k

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

def PrepareLstm(dsY):
    ds = dsY[0]
    # ds-->year, month, dayOfWeek 추출 #TODO

    size = len(dsY)
    year = np.random.beta(2000, 2017, size) * (2017 - 2000)
    month = np.random.beta(1, 12, size) * (12 - 1)
    dayOfWeek = np.random.beta(3, 6, size) * (6 - 0)
    y = dsY[1]
    # 이차원 배열
    preprocessedData=[year, month, dayOfWeek, y]
    return preprocessedData

def Lstm(preprocessedData,forecastDay):
    forecast=[]
    #일단은 random 출력
    min=1 #min(preprocessedData[-1])
    max=70 #max(preprocessedData[-1])
    return list(np.random.beta(min, max, forecastDay)*(max-min))

def PrepareBayseian(dsY):
    ds = dsY[0]
    y = np.log(dsY[1])
    sales = list(zip(ds, y))
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANpreprocessing dsY to list succeed", DefineManager.LOG_LEVEL_INFO)
    preprocessedData= pd.DataFrame(data = sales, columns=['ds', 'y'])
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANdataFrame succeed", DefineManager.LOG_LEVEL_INFO)
    return preprocessedData

def Bayseian(preprocessedData, forecastDay):
    forecast=[]
    model = Prophet()
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIAN forecast modelfit start",
                                   DefineManager.LOG_LEVEL_INFO)
    model.fit(preprocessedData)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIAN forecast modelfit success",
                                   DefineManager.LOG_LEVEL_INFO)
    future = model.make_future_dataframe(periods=forecastDay)
    forecast = future[-forecastDay:]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIAN forecast succeed", DefineManager.LOG_LEVEL_INFO)
    #forecastDay에 해당하는 date형식을 출력
    # (for firebase api: def StoreInputData(processId = 0, rawArrayData = [], rawArrayDate = [], day = 0))
    dateStamp = list(forecast['ds'][-forecastDay:])
    date = [p.strftime('%Y-%m-%d') for p in dateStamp]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIAN date retrieve succeed", DefineManager.LOG_LEVEL_INFO)
    return forecast, date
