import json

def UploadRawDatas():
    return json.dumps({"Result": 1})

def ForecastDatas():
    return json.dumps({"Status": "Done", "Result": []})