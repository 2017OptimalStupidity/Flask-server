from Settings import DefineManager
from Utils import LoggingManager
from . import FirebaseDatabaseManager
import pandas as pd
import numpy as np
from fbprophet import Prophet

def LearningModuleRunner(rawArrayDatas, processId, forecastDay):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    numOfAlgorithmModules=3
    rmse={}
    realForecast={}
    mockForecast={}
    trainSize=int(len(rawArrayDatas) * 0.7)
    testSize=len(rawArrayDatas)-trainSize

    dataY = rawArrayDatas[1]
    testY = dataY[trainSize:]
    ############################################################### LSTM
    # 전처리
        # [[날짜],[판매량]] 형태 2D list -> pandas.core.frame.DataFrame의 형태의 [년, 월, 요일, 판매량]
        #TODO ds를 요일, 계절로 변환을 PrepareLstm()에서 시행
        #TODO 날씨, 경제 등의 feature 도 pandas DatFrame에 포함(PrepareLstm()에서)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start", DefineManager.LOG_LEVEL_INFO)
        #전체 data 전처리
    XY=PrepareLstm(rawArrayDatas)
        #train data 추출(0.7)
    X = XY[0][:trainSize]
    Y = XY[1][:trainSize]
    trainXY=[X,Y]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "LSTMpreprocessing succeed", DefineManager.LOG_LEVEL_INFO)
    # 계산
        #trainXY로 testY 예측
    mockForecast['Lstm'] = Lstm(preprocessedData=trainXY, forecastDay=testSize)
        #전체 XY로 realForecast 예측
    realForecast['Lstm'] = Lstm(preprocessedData=XY, forecastDay=forecastDay)
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "LSTMpredict succeed", DefineManager.LOG_LEVEL_INFO)
    # 평가
    testRmse=0
    for i in range(testSize):
        testRmse=testRmse + (testY[i] - mockForecast['Lstm'][i]) ** 2
    rmse['Lstm'] = testRmse
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "LSTMevaluation succeed", DefineManager.LOG_LEVEL_INFO)

    ################################################################ BAYSEIAN
    # 전처리 raw data를 preprocessed data(ds-y)로 변환하는 과정
        #[[날짜],[판매량]] 형태 2D list -> pandas.core.frame.DataFrame의 형태의 [날짜, 판매량]

        # 전체 data 전처리
    XY = PrepareBayseian(rawArrayDatas)
        # train data 추출(0.7)
    X = XY[0][:trainSize]
    Y = XY[1][:trainSize]
    trainXY=[X,Y]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANpreprocessing succeed", DefineManager.LOG_LEVEL_INFO)
    #계산
        # trainXY로 testY 예측
    mockForecast['Bayseian'] = Bayseian(preprocessedData=trainXY, forecastDay=testSize)[0]
        # 전체 XY로 realForecast 예측
    realForecast['Bayseian'] = Bayseian(preprocessedData=XY, forecastDay=forecastDay)[0]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANpredict succeed", DefineManager.LOG_LEVEL_INFO)
    #평가
    testRmse=0
    for i in range(testSize):
        testRmse=testRmse+ (testY[i] - mockForecast['Bayseian'][i]) ** 2
    rmse['Bayseian'] = testRmse
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANevaluation succeed", DefineManager.LOG_LEVEL_INFO)
    ############################################################### 그 외 알고리즘(ex SVM)


######################################################################################################################
    # numOfAlgorithmModules 개의 결과값(mockForecast)을 다 취합해서 비교 후 가장 좋은 결과를 firebase에 저장
    # 가장 좋은 결과 평가기준: test set에 대한 <testY & mockForecast>의 rmse
    min = rmse['Lstm']
    bestAlgorithmName='Lstm'
    for i in rmse.keys():
        if(min>rmse[i]):
            min=rmse[i]
            bestAlgorithmName=i

    realForecast=realForecast[bestAlgorithmName]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "Algorithm comparison succeed", DefineManager.LOG_LEVEL_INFO)
    forecastDate= Bayseian(preprocessedData=XY, forecastDay=forecastDay)[1]
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "DateMaking succeed", DefineManager.LOG_LEVEL_INFO)
    FirebaseDatabaseManager.StoreOutputData(processId,forecastDate,realForecast,DefineManager.ALGORITHM_STATUS_DONE )
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "FirebaseUploading succeed", DefineManager.LOG_LEVEL_INFO)
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
    y = dsY[1]
    sales = list(zip(ds, y))
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "BAYSEIANpreprocessing dsY to list succeed", DefineManager.LOG_LEVEL_INFO)
    preprocessedData= pd.DataFrame(data = sales, columns=['ds', 'y'])
    return preprocessedData

def Bayseian(preprocessedData, forecastDay):
    forecast=[]
    model = Prophet()
    model.fit(preprocessedData)
    future = model.make_future_dataframe(periods=forecastDay)
    forecast = future[-forecastDay:]

    #forecastDay에 해당하는 date형식을 출력
    # (for firebase api: def StoreInputData(processId = 0, rawArrayData = [], rawArrayDate = [], day = 0))
    dateStamp = list(forecast['ds'][-forecastDay:])
    date = [p.strftime('%Y-%m-%d') for p in dateStamp]
    return forecast, date
