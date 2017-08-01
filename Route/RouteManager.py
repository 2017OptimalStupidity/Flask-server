from . import routes
from FrontEnd import HomePage
from Core import BackgroundProcessManager
from flask import request

@routes.route("/")
def IndexPage():
    return HomePage.RenderIndexPage()

@routes.route("/upload/", methods=['POST'])
def UploadRawDatas():
    content = request.get_json(silent=True)
    print (content)
    return BackgroundProcessManager.UploadRawDatas(content['Data'])

@routes.route("/forecast/", methods=['POST'])
def ForecastDatas():
    content = request.get_json(silent=True)
    print (content)
    return BackgroundProcessManager.ForecastDatas(content['ProcessId'], content['Day'])