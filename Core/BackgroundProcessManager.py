import json
import threading
from Utils import LoggingManager
from Settings import DefineManager
from . import LearningManager, FirebaseDatabaseManager

FirebaseDatabaseManager.GetFirebaseConnection(DefineManager.FIREBASE_DOMAIN)

def UploadRawDatas(rawDataArray, day):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "UploadRawDatas", "parameter: " + str(rawDataArray) + ", day " + str(day), DefineManager.LOG_LEVEL_INFO)

    queueId = AddNewTrain(rawDataArray, day)

    return json.dumps({"Result": queueId})

def ForecastDatas(processId):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "ForecastDatas", "parameter: id " + str(processId), DefineManager.LOG_LEVEL_INFO)

    forcastStatus, forecastedData = GetStoredTrain(processId)

    status = "Working"

    if forcastStatus == DefineManager.ALGORITHM_STATUS_DONE:
        status = "Done"
    else:
        forecastedData = []

    return json.dumps({"Status": status, "Result": forecastedData})

def AddNewTrain(rawDataArray, day):
    nowDictSize = FirebaseDatabaseManager.GetLastProcessId() + 1
    FirebaseDatabaseManager.CreateNewProcessTable(nowDictSize)
    FirebaseDatabaseManager.StoreInputData(nowDictSize, rawDataArray, day)
    FirebaseDatabaseManager.UpdateLastProcessId(nowDictSize)

    threadOfLearn = threading.Thread(target=LearningManager.LearningModuleRunner, args=(rawDataArray, nowDictSize, day))
    threadOfLearn.start()

    return nowDictSize

def GetStoredTrain(processId):

    processStatus = FirebaseDatabaseManager.GetOutputDataStatus(processId)
    processResult = FirebaseDatabaseManager.GetOutputDataArray(processId)
    #
    # if processStatus == DefineManager.ALGORITHM_STATUS_DONE:
    #     # LoggingManager.PrintLogMessage("Core", "GetStoredTrain", "dic: " + str(processingQueueDict[processId]), DefineManager.LOG_LEVEL_INFO)
    #     return processResult
    # else:
    #     return []
    return processStatus, processResult