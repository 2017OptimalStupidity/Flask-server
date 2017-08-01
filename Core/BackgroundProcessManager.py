import json

def UploadRawDatas(rawDataArray):
    return json.dumps({"Result": 1})

def ForecastDatas(processId, day):
    return json.dumps({"Status": "Done", "Result": []})