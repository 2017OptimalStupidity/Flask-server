import json
from Utils import LoggingManager
from Settings import DefineManager

def UploadRawDatas(rawDataArray):
    LoggingManager.PrintLogMessage("Core", "UploadRawDatas", "parameter: " + str(rawDataArray), DefineManager.LOG_LEVEL_INFO)
    return json.dumps({"Result": 1})

def ForecastDatas(processId, day):
    LoggingManager.PrintLogMessage("Core", "UploadRawDatas", "parameter: id " + str(processId) + ", day " + str(day), DefineManager.LOG_LEVEL_INFO)
    return json.dumps({"Status": "Done", "Result": []})