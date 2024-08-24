import pyaudio
import numpy as np
import torch
import torchaudio as ta
import torch.nn as nn
from gpiozero import LED, Button
import time
import os
import datetime
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import wave

#GPIO Pin Numers (BCM not Board Pins nums) 
relayChannel1 = 26 #Board pin 37, BCM pin 26
relayChannel2 = 20 #Board pin 38, BCM pin 20
relayChannel3 = 21 #Board pin 40, BCM pin 21
codecLEDGreen = 23
codecLEDRed   = 24
codecSwitch   = 27



greenLED = LED(codecLEDGreen)
redLED   = LED(codecLEDRed)
relayCh1 = LED(relayChannel1)
relayCh1.on() #Energize to turn off voltage to signal conditioner
codecButton = Button(codecSwitch)

primaryPath = "/home/dasl/repos/robot_rich/"



# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 96000
CHUNK = int(RATE * 0.025)  # 25ms chunk
INPUT_DEVICE_INDEX = 0 #Need to figure out what the index is on rPi

from_email = 'ashborerdetection@outlook.com'
to_email = 'jru34@nau.edu'
subject = 'Detected Events Report'


# SMTP server configuration for Outlook
smtp_server = 'smtp-mail.outlook.com'
smtp_port = 587
smtp_user = 'ashborerdetection@outlook.com'
smtp_password = pass
#Add security measures

def send_email(subject, body, to_email, from_email, attachments=[]):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    for file_path in attachments:
        attachment = MIMEBase('application', 'octet-stream')
        with open(file_path, 'rb') as f:
            attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename={file_path}')
        msg.attach(attachment)

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        print('Email sent!')


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

# Blink the LED for a short duration (e.g., 100ms on, 100ms off)
def flash_green_led():
    greenLED.blink(on_time=2, off_time=1, n=1, background=True)


def chunk_to_spec(audio_chunk):
    transform = ta.transforms.Spectrogram(n_fft = 128, hop_length=32, power = 1)
    y = transform(audio_chunk)
    return y

def classify_chunk(audio_chunk):
    spec = chunk_to_spec(audio_chunk)
    spec = spec.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(spec)
        pred = (output >= 0.5) * 1
    return pred.item()
def save_audio(frames, filename, channels=CHANNELS, rate=RATE, format=FORMAT):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f'Audio saved as {filename}')

def record_and_classify(duration_seconds=30):
    relayCh1.off() #Denergize to turn on voltage to signal conditioner
    print('Waiting 5 seconds to begin.')
    time.sleep(5) #give voltage in signal condition time to settle
    print('Beginning detection')
    p = pyaudio.PyAudio()

    # Open a stream for input
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,

                    )
    redLED.on()

    try:
        frames = []
        times = []
        num_chunks = int(duration_seconds * 1000 / 25)  # Total number of 25ms chunks
        for i in range(num_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_chunk = torch.from_numpy(audio_chunk.astype('f'))
            audio_chunk = audio_chunk/audio_chunk.max()
            audio_chunk = audio_chunk.unsqueeze(0)
            if i == 0:
                print(f'Chunk shape {audio_chunk.shape}')
            pred = classify_chunk(audio_chunk)

            if pred == 1:
                flash_green_led()
                print('event detected')
                start = int(i*.025*RATE)
                end = int((i+1)*.025*RATE)
                if i == 0:
                    print(f'incident length: {end-start}')
                times.append((start,end))

    except KeyboardInterrupt:
        print('Recording Interrupted')
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        relayCh1.on() #Denergize to turn on voltage to signal conditioner
        redLED.off()
        print('Recording Complete')
        if len(times) > 0:
            print(f'{len(times)} events detected.')
            audio_filename = 'recorded_audio.wav'
            save_audio(frames, audio_filename)
            print('Audio saved')
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)

            # Plot the full recording
            plt.figure(figsize=(12, 6))
            time_axis = np.arange(0, len(audio_data)) / RATE
            plt.plot(time_axis, audio_data, label='Audio Signal')
            for (start,end) in times:
                plt.plot(time_axis[start:end],audio_data[start:end],color='r')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Audio Signal with Detected Events')
            plt.legend()
            plt.grid(True)

            # Save the plot as a file
            plt.savefig('test_plot.png')  # Save the figure
            print(f'Plot saved')
            attachments = ['test_plot.png', audio_filename]
            print('Sending Email')
            body = f'{len(times)} events have been detected. See attached audio and plot.'
            send_email(subject, body, to_email, from_email, attachments)

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
    os.system("sudo alsactl restore -f "+primaryPath+setupPath+setupFilename)


   

if __name__ == "__main__":
    # Load your trained model
    model = CNNNetwork()
    model_path = '/home/dasl/repos/robot_rich/Best_96k_Label_Smoothed.pt'
    model.load_state_dict(torch.load(model_path, weights_only = True))  # Update this with the path to your model
    model.eval()
    recordingDuration = 10
    setdeviceparameters(4)
    record_and_classify(duration_seconds=recordingDuration)

