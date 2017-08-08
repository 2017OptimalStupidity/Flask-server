from firebase import firebase
from Utils import LoggingManager
from Settings import DefineManager

firebaseDatabase = None

def GetFirebaseConnection(firebaseAddress):
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

# https://i2max-project.firebaseio.com/
firebaseDatabase = GetFirebaseConnection('https://i2max-project.firebaseio.com/')
GetLastProcessId()
CloseFirebaseConnection()