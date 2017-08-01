from Settings import DefineManager
from Utils import LoggingManager

def RenderIndexPage():
    LoggingManager.PrintLogMessage("FrontEnd", "RenderIndexPage", "Print index page", DefineManager.LOG_LEVEL_INFO)
    return "Hello world"