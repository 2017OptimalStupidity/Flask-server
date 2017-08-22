from flask import Flask
from Route import *
import imp
import sys

imp.reload(sys)

app = Flask(__name__)
app.register_blueprint(routes)

if __name__ == '__main__':
	app.debug = True
	app.run(host=DefineManager.SERVER_USING_HOST, port=DefineManager.SERVER_USING_PORT)