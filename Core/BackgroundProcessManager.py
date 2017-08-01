import json
import threading
from Utils import LoggingManager
from Settings import DefineManager
from . import LearningManager

processingQueueDict = {}

def UploadRawDatas(rawDataArray, day):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "UploadRawDatas", "parameter: " + str(rawDataArray) + ", day " + str(day), DefineManager.LOG_LEVEL_INFO)

    queueId = AddNewTrain(rawDataArray, day)

    return json.dumps({"Result": queueId})

def ForecastDatas(processId):
    LoggingManager.PrintLogMessage("BackgroundProcessManager", "ForecastDatas", "parameter: id " + str(processId), DefineManager.LOG_LEVEL_INFO)

    forecastedData = GetStoredTrain(processId)

    return json.dumps({"Status": "Done", "Result": forecastedData})

def AddNewTrain(rawDataArray, day):
    nowDictSize = len(processingQueueDict)
    processingQueueDict[nowDictSize] = rawDataArray

    threadOfLearn = threading.Thread(target=LearningManager.LearningModuleRunner, args=(rawDataArray, nowDictSize, day))
    threadOfLearn.start()

    return nowDictSize

def GetStoredTrain(processId):

    processStatus = LearningManager.ProcessResultGetter(processId)[1]
    processResult = LearningManager.ProcessResultGetter(processId)[0]

    if processStatus == DefineManager.ALGORITHM_STATUS_DONE:
        # LoggingManager.PrintLogMessage("Core", "GetStoredTrain", "dic: " + str(processingQueueDict[processId]), DefineManager.LOG_LEVEL_INFO)
        return processResult
    else:
        return []