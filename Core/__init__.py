from . import FirebaseDatabaseManager
from Settings import DefineManager

FirebaseDatabaseManager.GetFirebaseConnection(DefineManager.FIREBASE_DOMAIN)

lastProcessIdNumber = FirebaseDatabaseManager.GetLastProcessId()