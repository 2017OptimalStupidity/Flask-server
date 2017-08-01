from . import routes
from FrontEnd import HomePage

@routes.route("/")
def IndexPage():
    return HomePage.RenderIndexPage()