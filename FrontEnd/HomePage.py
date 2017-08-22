from Settings import DefineManager
from Utils import LoggingManager
from flask import render_template

def RenderIndexPage():
    LoggingManager.PrintLogMessage("FrontEnd", "RenderIndexPage", "Print index page", DefineManager.LOG_LEVEL_INFO)
    return render_template('index.html')

def MailContect(name = "Anonymous", email = "Anonymous@anonymous.com", message = "No message data"):
    LoggingManager.PrintLogMessage("FrontEnd", "MailContect", "Sending email name: " + name + " email: " + email + " msg: " + message, DefineManager.LOG_LEVEL_INFO)