import json
import threading
from Utils import LoggingManager
from Settings import DefineManager
from . import LearningManager, FirebaseDatabaseManager

FirebaseDatabaseManager.GetFirebaseConnection(DefineManager.FIREBASE_DOMAIN)

def UploadRawDatas(rawDataArray, rawDateArray, day):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "UploadRawDatas", "data: " + str(rawDataArray) + ", date: " + str(rawDateArray) + ", day " + str(day), DefineManager.LOG_LEVEL_INFO)

    queueId = AddNewTrain(rawDataArray, rawDateArray, day)

    return json.dumps({"Result": queueId})

def ForecastDatas(processId):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "ForecastDatas", "parameter: id " + str(processId), DefineManager.LOG_LEVEL_INFO)

    forcastStatus, forecastedData = GetStoredTrain(processId)

    status = "Working"

    if forcastStatus == DefineManager.ALGORITHM_STATUS_DONE:
        status = "Done"
        return json.dumps({"Status": status, "Result": forecastedData[DefineManager.DATA_SAVED_POINT], "Date": forecastedData[DefineManager.DATE_SAVED_POINT]})
    else:
        forecastedData = []
        return json.dumps({"Status": status, "Result": forecastedData, "Date": []})


def AddNewTrain(rawDataArray, rawDateArray, day):
    nowDictSize = FirebaseDatabaseManager.GetLastProcessId() + 1
    FirebaseDatabaseManager.CreateNewProcessTable(nowDictSize)
    FirebaseDatabaseManager.StoreInputData(nowDictSize, rawDataArray, rawDateArray, day)
    FirebaseDatabaseManager.UpdateLastProcessId(nowDictSize)

    rawDataAndDateArray = [rawDateArray, rawDataArray]
    threadOfLearn = threading.Thread(target=LearningManager.LearningModuleRunner, args=(rawDataAndDateArray, nowDictSize, day))
    threadOfLearn.start()

    return nowDictSize

def GetStoredTrain(processId):

    processResult = LearningManager.ProcessResultGetter(processId)

    processStatus = processResult[1]
    processData = processResult[0]

    # processStatus = FirebaseDatabaseManager.GetOutputDataStatus(processId)
    # processResult = FirebaseDatabaseManager.GetOutputDataArray(processId)
    #
    # if processStatus == DefineManager.ALGORITHM_STATUS_DONE:
    #     # LoggingManager.PrintLogMessage("Core", "GetStoredTrain", "dic: " + str(processingQueueDict[processId]), DefineManager.LOG_LEVEL_INFO)
    #     return processResult
    # else:
    #     return []
    return processStatus, processData