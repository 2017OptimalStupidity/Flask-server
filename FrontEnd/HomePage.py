from Settings import DefineManager
from Utils import LoggingManager
from flask import render_template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

def RenderIndexPage():
    LoggingManager.PrintLogMessage("FrontEnd", "RenderIndexPage", "Print index page", DefineManager.LOG_LEVEL_INFO)
    return render_template('index.html')

def MailContect(name = "Anonymous", email = "Anonymous@anonymous.com", message = "No message data"):
    LoggingManager.PrintLogMessage("FrontEnd", "MailContect", "Sending email name: " + name + " email: " + email + " msg: " + message, DefineManager.LOG_LEVEL_INFO)

    if name == None or email == None or message == None or name == "" or email == "" or message == "":
        return DefineManager.NOT_AVAILABLE

    try:
        emailReceiveManager = "*******@gmail.com"

        fromEmailAddr = email
        toEmailAddr = emailReceiveManager
        msg = MIMEMultipart()
        msg['From'] = fromEmailAddr
        msg['To'] = toEmailAddr
        msg['Subject'] = name + " send message"

        msg.attach(MIMEText(message, 'plain'))
        LoggingManager.PrintLogMessage("FrontEnd", "MailContect", "email rdy to send", DefineManager.LOG_LEVEL_INFO);
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(emailReceiveManager, "********")
        text = msg.as_string()

        server.sendmail(fromEmailAddr, toEmailAddr, text)
        LoggingManager.PrintLogMessage("FrontEnd", "MailContect", "mail sent", DefineManager.LOG_LEVEL_INFO);
        return DefineManager.AVAILABLE
    except:
        LoggingManager.PrintLogMessage("FrontEnd", "MailContect", "Email sending error!", DefineManager.LOG_LEVEL_ERROR)
        return DefineManager.NOT_AVAILABLE
