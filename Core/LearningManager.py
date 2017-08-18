from Settings import DefineManager
from Utils import LoggingManager
from . import FirebaseDatabaseManager
import pandas as pd
import numpy as np
from fbprophet import Prophet

def LearningModuleRunner(rawArrayDatas, processId, forecastDay):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    mockForecast={}
    rmse={}
    forecast=[]
    realForecast={}

    trainSize=int(len(rawArrayDatas[0])*0.7)
    testSize=len(rawArrayDatas[0])-trainSize

    ds=rawArrayDatas[0]
    y=list(np.log(rawArrayDatas[1]))
    sales=list(zip(ds, y))

    preprocessedData=pd.DataFrame(data=sales, columns=['ds','y'])
    model=Prophet()
    model.fit(preprocessedData)

    future = model.make_future_dataframe(periods=forecastDay)
    forecast = future[-forecastDay:]
    forecast = model.predict(future)
    forecastData=[np.exp(y) for y in forecast['yhat'][-forecastDay:]]
    data=rawArrayDatas[1]+forecastData
    date = [d.strftime('%Y-%m-%d') for d in forecast['ds']]

    FirebaseDatabaseManager.StoreOutputData(processId, resultArrayData=data, resultArrayDate=date, status=DefineManager.ALGORITHM_STATUS_DONE)
    return

def ProcessResultGetter(processId):#TODO: Request processid:2 -> {"Result": [3, 4], "Status": "Done", "Date": null}

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
    dayOfWeek = np.random.beta(3, 6, size) * (6 - 0)#TODO: ValueError: a <= 0, KeyError: 0
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
