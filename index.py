from flask import Flask
from Settings import DefineManager
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
	app.debug = True
	app.run(host=DefineManager.SERVER_USING_HOST, port=DefineManager.SERVER_USING_PORT)