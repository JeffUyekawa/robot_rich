#!/bin/bash
#use sudo chmod +x /home/dasl/code/audioRecorder/recorderLaunch.sh
#in terminal to make this file executable. Otherwise you'll get
#a permission denied error.

source /home/dasl/repos/robot_rich/.mlvenv/bin/activate
sudo python3 /home/dasl/repos/robot_rich/codecrec.py 10 30
