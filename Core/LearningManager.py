from Settings import DefineManager
from Utils import LoggingManager

def LearningModuleRunner(rawArrayDatas, processId, day):
    LoggingManager.PrintLogMessage("LearningManater", "LearningModuleRunner", "start of learning #" + str(processId), DefineManager.LOG_LEVEL_INFO)
    return

def ProcessResultGetter(processId):
    return [[], DefineManager.ALGORITHM_STATUS_WORKING]