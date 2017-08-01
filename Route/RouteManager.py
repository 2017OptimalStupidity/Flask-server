from . import routes

@routes.route("/")
def HelloWorld():
    return "Hello world"