#!/bin/bash

echo "Killing all python program"
killall python3.5

echo "Pulling latest server program"
git pull https://github.com/I2MAX-LearningProject/Flask-server.git

echo "Starting python server"
nohup python3.5 -u index.py &
