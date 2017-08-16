import requests
import json

CLOUD_END_POINT = "http://211.249.49.198:5000"

def test_IsHomePageWorking():
    homePageRequest = requests.get(CLOUD_END_POINT)
    assert homePageRequest.text == "Hello world"

def test_UploadTestData():
    testUploadData = {
        "Data": [20.0, 30.0, 401.0, 50.0, 60.0],
        "Date": ["2017-08-11", "2017-08-12", "2017-08-13", "2017-08-14", "2017-08-15"],
        "Day": 4
    }
    uploadTestDataRequest = requests.post(CLOUD_END_POINT + "/upload/", json=testUploadData)

    serverResponse = json.loads(uploadTestDataRequest.text)
    assert serverResponse["Result"] >= 0, "Wrong process id returned"
    return

def test_DownloadWorkingForecastData():
    testUploadData = {
        "ProcessId": 9
    }
    uploadTestDataRequest = requests.post(CLOUD_END_POINT + "/forecast/", json=testUploadData)

    serverResponse = json.loads(uploadTestDataRequest.text)
    assert serverResponse["Status"] == "Working", "Is it really process done?"

def test_DownloadDoneForecastData():
    testUploadData = {
        "ProcessId": 2
    }
    uploadTestDataRequest = requests.post(CLOUD_END_POINT + "/forecast/", json=testUploadData)

    serverResponse = json.loads(uploadTestDataRequest.text)
    assert serverResponse["Status"] == "Done", "That process id must return status done"