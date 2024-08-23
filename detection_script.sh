#!/bin/bash
#use sudo chmod +x /home/dasl/code/audioRecorder/recorderLaunch.sh
#in terminal to make this file executable. Otherwise you'll get
#a permission denied error.

source /home/dasl/repos/robot_rich/.mlvenv/bin/activate
sudo /home/dasl/repos/robot_rich/.mlvenv/bin/python /home/dasl/repos/robot_rich/real_time_detection.py
