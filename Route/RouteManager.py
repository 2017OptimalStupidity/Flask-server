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
    return BackgroundProcessManager.UploadRawDatas(content['Data'], content['Date'], content['Day'])

@routes.route("/forecast/", methods=['POST'])
def ForecastDatas():
    content = request.get_json(silent=True)
    LoggingManager.PrintLogMessage("RouteManager", "ForecastDatas", "json data: " + str(content), DefineManager.LOG_LEVEL_INFO)
    return BackgroundProcessManager.ForecastDatas(content['ProcessId'])

@routes.route("/mail/", methods=['POST'])
def SendMail():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    emailStatus = HomePage.MailContect(name, email, message)

    if emailStatus == DefineManager.AVAILABLE:
        return "<script>alert('ok');location.href='/';</script>"
    else:
        return "<script>alert('fail');location.href='/';</script>"