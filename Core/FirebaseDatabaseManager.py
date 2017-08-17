from firebase import firebase
from Utils import LoggingManager
from Settings import DefineManager

firebaseDatabase = None

def GetFirebaseConnection(firebaseAddress = ""):
    global firebaseDatabase

    LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetFirebaseConnection", "getting firebase connection", DefineManager.LOG_LEVEL_INFO)

    if IsConnectionAlive() != True:
        try:
            firebaseDatabase = firebase.FirebaseApplication(firebaseAddress, None)
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetFirebaseConnection", "connection successful", DefineManager.LOG_LEVEL_INFO)
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetFirebaseConnection", "connection failure", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetFirebaseConnection", "you already connected", DefineManager.LOG_LEVEL_WARN)
    return firebaseDatabase

def CloseFirebaseConnection():
    global firebaseDatabase

    LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CloseFirebaseConnection", "closing firebase connection", DefineManager.LOG_LEVEL_INFO)

    if IsConnectionAlive():
        firebaseDatabase = None
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CloseFirebaseConnection", "connection closed", DefineManager.LOG_LEVEL_INFO)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CloseFirebaseConnection", "connection already closed", DefineManager.LOG_LEVEL_WARN)

def IsConnectionAlive():
    global firebaseDatabase
    if firebaseDatabase != None:
        return True
    else:
        return False

def GetLastProcessId():
    global firebaseDatabase

    if IsConnectionAlive():
        lastProcessId = firebaseDatabase.get('/lastProcessId', None)
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetLastProcessId", "last process id: " + str(lastProcessId), DefineManager.LOG_LEVEL_INFO)
        return lastProcessId
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetLastProcessId", "connection dead", DefineManager.LOG_LEVEL_WARN)
        return DefineManager.NOT_AVAILABLE

def UpdateLastProcessId(lastProcessId = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            postResult = firebaseDatabase.patch('/', {'lastProcessId': lastProcessId})
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "UpdateLastProcessId", "last process id updated: " + str(lastProcessId), DefineManager.LOG_LEVEL_INFO)
            return postResult
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "UpdateLastProcessId", "there is problem to post data", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "UpdateLastProcessId", "connection dead", DefineManager.LOG_LEVEL_WARN)
        return DefineManager.NOT_AVAILABLE

def CreateNewProcessTable(processId = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            postResult = firebaseDatabase.patch('/', {str(processId): {'inputData': {'data': [], 'date': [], 'day': 0}, 'outputData': {'data': [], 'date': [], 'status': DefineManager.ALGORITHM_STATUS_WORKING}}})
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CreateNewProcessTable", "creating new table", DefineManager.LOG_LEVEL_INFO)
            return True
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CreateNewProcessTable", "there is problem to create table", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "CreateNewProcessTable", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return False

def StoreInputData(processId = 0, rawArrayData = [], rawArrayDate = [], day = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            postResult = firebaseDatabase.patch('/' + str(processId) + '/inputData', {'data': rawArrayData, 'date': rawArrayDate, 'day': day})
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreInputData", "saved data", DefineManager.LOG_LEVEL_INFO)
            return True
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreInputData", "there is problem to store input data", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreInputData", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return False

def StoreOutputData(processId = 0, resultArrayData = [], resultArrayDate = [], status = DefineManager.ALGORITHM_STATUS_WORKING):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            postResult = firebaseDatabase.patch('/' + str(processId) + '/outputData', {'data': resultArrayData, 'date': resultArrayDate, 'status': status})
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreOutputData", "saved data", DefineManager.LOG_LEVEL_INFO)
            return True
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreOutputData", "there is problem to store output data", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "StoreOutputData", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return False

def GetOutputDataStatus(processId = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            outputStatus = firebaseDatabase.get('/' + str(processId) + '/outputData/status', None)
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "data status loaded: " + str(outputStatus), DefineManager.LOG_LEVEL_INFO)
            return outputStatus
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "there is problem to load status", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return DefineManager.NOT_AVAILABLE

def GetOutputDataArray(processId = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            outputArray = firebaseDatabase.get('/' + str(processId) + '/outputData/data', None)
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "data loaded", DefineManager.LOG_LEVEL_INFO)

            if outputArray == None:
                return []
            return outputArray
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "there is problem to load status", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return []

def GetOutputDateArray(processId = 0):
    global firebaseDatabase

    if IsConnectionAlive():
        try:
            outputArray = firebaseDatabase.get('/' + str(processId) + '/outputData/date', None)
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "data loaded", DefineManager.LOG_LEVEL_INFO)

            if outputArray == None:
                return []
            return outputArray
        except:
            LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "there is problem to load status", DefineManager.LOG_LEVEL_ERROR)
    else:
        LoggingManager.PrintLogMessage("FirebaseDatabaseManager", "GetOutputDataStatus", "connection dead", DefineManager.LOG_LEVEL_WARN)
    return []

# GetFirebaseConnection(DefineManager.FIREBASE_DOMAIN)
# GetLastProcessId()
# UpdateLastProcessId(5)
# CreateNewProcessTable(2)
# StoreInputData(2, [1, 2, 3], 2)
# StoreOutputData(2, [3, 4], DefineManager.ALGORITHM_STATUS_DONE)
# GetOutputDataStatus(2)
# GetOutputDataArray(2)
# CloseFirebaseConnection()
