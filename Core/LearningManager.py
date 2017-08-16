from Settings import DefineManager
from Utils import LoggingManager
from . import FirebaseDatabaseManager
import pandas as pd
import numpy as np
from fbprophet import Prophet

def LearningModuleRunner(rawArrayDatas, processId, day):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    numOfAlgorithmModules=3
    rmse={}
    result={}
    test_forecast={}
    train_size=int(len(rawArrayDatas) * 0.7)
    test_size=len(rawArrayDatas)-train_size

    dataY = rawArrayDatas[1]
    testY = dataY[train_size:]
    ############################################################### LSTM
    # raw data를 preprocessed data로 변환하는 과정
        # [[날짜],[판매량]] 형태로 되어있는 이차원 배열을 pandas.core.frame.DataFrame의 형태로 변환
        # 1. ds추가가공 통해 요일, 계절로 변환(함수)
        # 2. [년, 월, 요일, 판매량]
        # 3. 추후) + 날씨, 경제 등 (함수)

    XY=PrepareLstm(rawArrayDatas)
    X=XY[0][:train_size]
    Y = XY[1][:train_size]
    trainXY=[X,Y]
    #trainXY는 XY에서 전체 행의 0.7 자른 것

    # 계산
    test_forecast['Lstm'] = Lstm(preprocessedData=trainXY, forecastDay=test_size)
    result['Lstm'] = Lstm(preprocessedData=XY, forecastDay=day)

    #평가
    testRmse=0
    for i in range(test_size):
        testRmse=testRmse+(testY[i]-test_forecast['Lstm'][i])**2
    rmse['Lstm'] = testRmse

    ################################################################ BAYSEIAN

    #raw data를 preprocessed data(ds-y)로 변환하는 과정
         #[[날짜],[판매량]] 형태로 되어있는 이차원 배열을 pandas.core.frame.DataFrame의 형태로 변환

    XY = PrepareBayseian(rawArrayDatas)
    X=XY[0][:train_size]
    Y = XY[1][:train_size]
    trainXY=[X,Y]
    #trainXY는 XY에서 전체 행의 0.7 자른 것
    #계산
    test_forecast['Bayseian'] = Bayseian(preprocessedData=trainXY, forecastDay=test_size)[0]
    result['Bayseian'] = Bayseian(preprocessedData=XY, forecastDay=day)[0]

    #평가
    testRmse=0
    for i in range(test_size):
        testRmse=testRmse+(testY[i]-test_forecast['BAYSEIAN'][i])**2
    rmse['Bayseian'] = testRmse

    ############################################################### 그 외 (ex SVM)
    # 1. ds-y

######################################################################################################################
    # 세개에서 나온 결과값을 다 취합해서 비교해서 가장 좋은 결과를 firebase에 저장
    # 가장 좋은 결과의 평가기준은 예측값과 test set의 rmse
    min = rmse['Lstm']
    for i in rmse.keys():
        if(min>rmse[i]):
            min=rmse[i]
            bestAlgorithmName=i

    result=result[bestAlgorithmName]
    #processId = 0, resultArrayData = [], resultArrayDate = [], status = DefineManager.ALGORITHM_STATUS_WORKING)
    forecastDate= Bayseian(preprocessedData=XY, forecastDay=day)[1]
    FirebaseDatabaseManager.StoreOutputData(processId,forecastDate,result,DefineManager.ALGORITHM_STATUS_DONE )
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

    size=len(dsY)
    year = np.random.beta(2000, 2017, size) * (2017 - 2000)
    month = np.random.beta(1, 12, size) * (12 - 1)
    dayOfWeek = np.random.beta(3, 6, size) * (6 - 0)#TODO: ValueError: a <= 0, KeyError: 0
    y = dsY[1]
    # 이차원 배열
    preprocessedData=[year, month, dayOfWeek, y]
    return preprocessedData


def PrepareBayseian(dsY):
    ds = dsY[0]
    y = dsY[1]
    sales = list(zip(ds, y))
    preprocessedData= pd.DataFrame(data = sales, columns=['ds', 'y'])
    return preprocessedData


def Lstm(preprocessedData,forecastDay):
    forecast=[]
    #일단은 random 출력
    min=1 #min(preprocessedData[-1])
    max=70 #max(preprocessedData[-1])
    return list(np.random.beta(min, max, forecastDay)*(max-min))

def Bayseian(preprocessedData,forecastDay):
    forecast=[]
    model = Prophet()
    model.fit(preprocessedData)
    future = model.make_future_dataframe(periods=forecastDay)

    forecast=future[-forecastDay:]
    temp = list(forecast['ds'][-forecastDay:])
    date = [p.strftime('%Y-%m-%d') for p in temp]
    return forecast, date
