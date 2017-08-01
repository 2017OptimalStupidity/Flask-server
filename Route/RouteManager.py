from . import routes
from FrontEnd import HomePage
from Core import BackgroundProcessManager
from flask import request
from Utils import LoggingManager
from Settings import DefineManager

@routes.route("/")
def IndexPage():
    LoggingManager.PrintLogMessage("RouteManager", "IndexPage", "web page connection!", DefineManager.LOG_LEVEL_INFO)
    return HomePage.RenderIndexPage()

@routes.route("/upload/", methods=['POST'])
def UploadRawDatas():
    content = request.get_json(silent=True)
    LoggingManager.PrintLogMessage("RouteManager", "UploadRawDatas", "json data: " + str(content), DefineManager.LOG_LEVEL_INFO)
    return BackgroundProcessManager.UploadRawDatas(content['Data'])

@routes.route("/forecast/", methods=['POST'])
def ForecastDatas():
    content = request.get_json(silent=True)
    LoggingManager.PrintLogMessage("RouteManager", "ForecastDatas", "json data: " + str(content), DefineManager.LOG_LEVEL_INFO)
    return BackgroundProcessManager.ForecastDatas(content['ProcessId'], content['Day'])