import json
import threading
from Utils import LoggingManager
from Settings import DefineManager

processingQueueDict = {}

def UploadRawDatas(rawDataArray):
    LoggingManager.PrintLogMessage("Core", "UploadRawDatas", "parameter: " + str(rawDataArray), DefineManager.LOG_LEVEL_INFO)

    queueId = AddNewTrain(rawDataArray)

    return json.dumps({"Result": queueId})

def ForecastDatas(processId, day):
    LoggingManager.PrintLogMessage("Core", "UploadRawDatas", "parameter: id " + str(processId) + ", day " + str(day), DefineManager.LOG_LEVEL_INFO)

    forecastedData = GetStoredTrain(processId, day)

    return json.dumps({"Status": "Done", "Result": forecastedData})

def AddNewTrain(rawDataArray):
    nowDictSize = len(processingQueueDict)
    processingQueueDict[nowDictSize] = rawDataArray
    return nowDictSize

def GetStoredTrain(processId, day):
    if "Done" == DefineManager.ALGORITHM_STATUS_DONE:
        return processingQueueDict[processId]
    else:
        return []