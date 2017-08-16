from Settings import DefineManager
from Utils import LoggingManager
import pandas as pd
import numpy as np
from fbprophet import Prophet

def LearningModuleRunner(rawArrayDatas, processId, day):
    LoggingManager.PrintLogMessage("LearningManager", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)

    numOfAlgorithmModules=3
    rmse={} #dictionary
    result={} #dictionary
    test_forecast={}
    train = rawArrayDatas[:int(len(rawArrayDatas) * 0.7)]
    test = rawArrayDatas[int(len(rawArrayDatas) * 0.7):]

    # LSTM
    # 1. ds-y
    # 2 ds추가가공 통해 요일, 계절 등 처리(함수)
    # 3. 날씨, 경제 등 (함수)

    # day길이의  배열저장

    # 계산계산계산
    forecastDay=len(test)



    # 두개 출력(test를 위한 것, 실제 예측)
    test_forecast['LSTM']
    result['LSTM']=[]

    testRmse=0
    for i in range(len(test)):
        testRmse=testRmse+(test[i]-test_forecast['LSTM'][i])**2
    rmse['LSTM'] = testRmse





    # BAYSEIAN
    # 1. ds-y
    result['BAYSEIAN'] = []
    rmse['BAYSEIAN'] = testRmse

    test_forecast['BAYSEIAN']
    result['BAYSEIAN']=[]

    testRmse=0
    for i in range(len(test)):
        testRmse=testRmse+(test[i]-test_forecast['BAYSEIAN'][i])**2
    rmse['BAYSEIAN'] = testRmse








    # 그 외 (ex SVM)
    # 1. ds-y







    # 세개에서 나온 결과값을 다 취합해서 비교해서 가장 좋은 결과를 firebase에 저장
    # 가장 좋은 결과의 평가기준은 예측값과 test set의 rmse
    min = rmse['LSTM']
    for i in result.keys():
        if(min>rmse[i]):
            min=rmse[i]
            z=i
    firebase.store(result[z])
    return

def ProcessResultGetter(processId):
    if(DefineManager.ALGORITHM_STATUS_WORKING):
        return [[], DefineManager.ALGORITHM_STATUS_WORKING]
    if(DefineManager.ALGORITHM_STATUS_DONE):
        return [[],[]], DefineManager.ALGORITHM_STATUS_DONE


def Bayseian(preprocessedData,forecastDay):
    m = Prophet()
    m.fit(preprocessedData);
    future = m.make_future_dataframe(periods=forecastDay)

    test_forecast=future[-forecastDay:]
    for i in range(len(test)):
        testRmse=testRmse+(test[i]-test_forecast['BAYSEIAN'][i])**2
    rmse['BAYSEIAN'] = testRmse
