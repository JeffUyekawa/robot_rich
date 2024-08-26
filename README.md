# boring_recorder
An RPI audio recorder

### Hardware
This system expects the [RPI Relay Board](https://www.waveshare.com/wiki/RPi_Relay_Board) and the [CodecZero](https://www.raspberrypi.com/documentation/accessories/audio.html) to be installed on the RPI. The relay board should be wired to provide 27 V to the PCB signal conditioner as shown in the wiring image in the repo. Also, it is critical that the [RTC battery be installed and the RPI set up to charge the battery](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#real-time-clock-rtc).

### Pre-reqs
1. Create a repos directory on your RPI in your home folder:
   ```
   user@raspberrypi:~ $ mkdir repos
   user@raspberrypi:~ $ cd repos
   ```
2. Clone this repo into that directory:
   ```
   git clone https://github.com/dynamic-and-active-systems-lab/boring_recorder
   ```
3. Pi-Codec is a submodule that can be initialized after the cloning of this repo with 
	`git submodule update --init --recursive`

4. When using this for the first time, you need to generate the virtual environment and install the necessary packages:
  ```
	python3 -m venv .venv
	source ./venv/bin/activate
	pip install -r requirements.txt
  ```
### Running on boot
To get the audio recording to run on boot, you need to edit the crontab file on the RPI:
```
sudo crontab -e
```
Once in the file add the following:
```
@reboot /bin/sleep 10; /home/dasl/repos/boring_recorder/recorderLaunch.sh  >> /home/dasl/repos/boring_recorder/mycrontablog.txt 2>&1
```

### Notes on modifications of python packages or submodules
1. If you add new package requirements, add them to the requirements.txt file using
	`pip3 freeze > requirement.txt`
2. Note on initial setup: to add other submodules, use
`git submodule add <webaddress_of_git_repo>`


### Wiring image
![wiring_image](https://github.com/dynamic-and-active-systems-lab/boring_recorder/blob/main/wiring_setup.jpg)

### Notes on RPI Pins Usage
The relay board documentation can be found [here](https://www.waveshare.com/wiki/RPi_Relay_Board). It states states that the communication occurs on RPI pins 37, 38, and 40 (BCM no. 26, 20, and 21, respectively). 

The Codec Zero board documentation can be foudn [here](https://cdn.shopify.com/s/files/1/0174/1800/files/iqaudio-product-brief.pdf?v=1607939668). It states  that the communication occurs on RPI pins 3, 5, 12, 35, 38, 40 (GPIO 2, 3, 18, 19, 20, and 21). 

There is a conflict of RPI Pins 38 and 40 between these two boards. The scripts in this repo currently only use the pin 37 relay channel to turn on and off the peripheral signal conditioner. The using the other channels have not been tested and could cause problems. This is noted in the code. 

### Notes on ML Detection and Email Functionality
The device is set up to send an email using an email from a microsoft outlook account using a smpt server. 

If running for the first time, make a text file called email_info.txt with "from email" "to email" "from email password" as its contents.


### Remote Access
1. Connect to the same wifi network as the device.
2. (Optional) use a scanner to search IP addresses. (Suggested: Fing)
3. ssh in with
```
ssh dasl@ipaddress
```
4. If vnc-viewer is desired, use
```
vncserver-virtual
```
5. A new address will be printed that can be used with RealVNCViewer
