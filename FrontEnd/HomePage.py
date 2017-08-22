from Settings import DefineManager
from Utils import LoggingManager
from flask import render_template

def RenderIndexPage():
    LoggingManager.PrintLogMessage("FrontEnd", "RenderIndexPage", "Print index page", DefineManager.LOG_LEVEL_INFO)
    return render_template('index.html')