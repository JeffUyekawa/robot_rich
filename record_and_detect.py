#import sounddevice as sd
#from scipy.io.wavfile import write
#import wavio as wv
#from sense_hat import SenseHat
import os
import time
from datetime import datetime
from gpiozero import LED
from gpiozero import Button
import sys
import torchaudio as ta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

#sense = SenseHat()
#GPIO Pin Numers (BCM not Board Pins nums) 
relayChannel1 = 26 #Board pin 37, BCM pin 26
relayChannel2 = 20 #Board pin 38, BCM pin 20
relayChannel3 = 21 #Board pin 40, BCM pin 21
codecLEDGreen = 23
codecLEDRed   = 24
codecSwitch   = 27

haltFlag = False

greenLED = LED(codecLEDGreen)
redLED   = LED(codecLEDRed)
relayCh1 = LED(relayChannel1)
relayCh1.on() #Energize to turn off voltage to signal conditioner
codecButton = Button(codecSwitch)

primaryPath = "/home/dasl/repos/robot_rich/"

# 
class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv7=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
       
       
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=3072,out_features=128)
        self.linear2=nn.Linear(in_features=128,out_features=1)
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output = self.output(logits)
       
        
        return output
def chunk_to_spec(audio_chunk):
    transform = ta.transforms.Spectrogram(n_fft = 128, hop_length=32, power = 1)
    y = transform(audio_chunk)
    return y

def classify_chunk(audio_chunk,model):
    spec = chunk_to_spec(audio_chunk)
    spec = spec.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(spec)
        pred = (output >= 0.5) * 1
    return pred.item()
   
def makewavefile(fs = 96000, duration=5, channels = 1, filename="test.wav"):
    relayCh1.off() #Denergize to turn on voltage to signal conditioner
    time.sleep(5) #give voltage in signal condition time to settle
    print('Recording')
    redLED.on()
    recordString = "arecord -D 'plughw:CARD=IQaudIOCODEC,DEV=0' -t wav -r "+str(int(fs))+" -c "+str(int(channels))+" -f S24_LE -d " + str(duration) + " " + filename
    os.system(recordString)
    print(recordString)
    relayCh1.on() #Denergize to turn on voltage to signal conditioner
    redLED.off()
    print('Done.')

def setdeviceparameters(fileNum=1):
    setupPath = "Pi-Codec/"
    setupFilename1 = "Codec_Zero_AUXIN_record_and_HP_playback.state"
    setupFilename2 = "Codec_Zero_OnboardMIC_record_and_SPK_playback.state"
    setupFilename3 = "Codec_Zero_Playback_only.state"
    setupFilename4 = "Codec_Zero_StereoMIC_record_and_HP_playback.state"
    if fileNum==1:
        setupFilename = setupFilename1
    elif fileNum==2:
        setupFilename = setupFilename2
    elif fileNum==3:
        setupFilename = setupFilename3
    elif fileNum==4:
        setupFilename = setupFilename4
    cwdir=os.getcwd()
    print(cwdir)
    os.system("sudo alsactl restore -f "+primaryPath+setupPath+setupFilename)

def simulate_real_time_classification(model,wav_file, duration = 10):
    audio_data, sr = ta.load(wav_file)
    print('loading audio')
    if audio_data.shape[0] > 1:
        audio_data = audio_data[0,:].reshape(1,-1)
    
    if sr != 96000:
        resampler = ta.transforms.Resample(sr,96000)
        start = int(5*sr)
        audio_data = audio_data[:,start:]
        audio_data = resampler(audio_data)
        sr = 96000
    num_chunks = int(duration * 1000/25)
    times_list = []
    try:
        for i in range(num_chunks):
            start_idx = i*int(0.025*sr)
            end_idx = start_idx + int(0.025*sr)
            if end_idx > audio_data.shape[1]:
                break
            audio_chunk = audio_data[:,start_idx:end_idx]
            audio_chunk = audio_chunk/audio_chunk.max()
        
            pred = classify_chunk(audio_chunk,model)
            
            if pred== 1:
                times_list.append((start_idx,end_idx))
    finally:
        pass
        
    return times_list

if __name__=="__main__":
    recordingDir = "/home/dasl/repos/robot_rich"
    recordingDuration = 10
 
    nowTimeStamp = datetime.now()
    nowTimeStampStr = nowTimeStamp.strftime("%Y-%m-%d_%H_%M_%S")
    fileName = "test.wav"
    # redLED.on()
    # time.sleep(5)
    # redLED.off()
    fullPathName = recordingDir+"/"+fileName
    setdeviceparameters(4)
    makewavefile(96000, recordingDuration, 1, fullPathName)
    time.sleep(10)
    y, fs = ta.load(fullPathName)
    print('Starting Detection')
    print(f'File used {fullPathName}')
    t = np.arange(y.shape[1])/fs
    model = CNNNetwork()
    model.load_state_dict(torch.load('/home/dasl/repos/robot_rich/Best_96k_Label_Smoothed.pt',weights_only=True))
    times = simulate_real_time_classification(model,fullPathName)
    plt.figure(figsize=(12, 6))
    y = y/y.max()
    plt.plot(t,y[0].numpy())
    for (start,end) in times:
        plt.plot(t[start:end],y[0,start:end].numpy(),color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal with Detected Events')
    plt.legend()
    plt.grid(True)

    # Save the plot as a file
    plt.savefig('test_plot_2.png')  # Save the figure
    print(f'Plot saved')


    time.sleep(5)
    
