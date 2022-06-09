#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys
import time
import audioop

CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

p = pyaudio.PyAudio()

device = p.get_default_output_device_info()
#dev_samplerate = int(device['defaultSampleRate'])
dev_samplerate = 16000
dev_index = device['index']

print("Device Info: %r" % device)
print("")

filename = sys.argv[1]
wf = wave.open(filename, 'rb')

wf_channels = wf.getnchannels()
wf_sampwidth = wf.getsampwidth()
wf_format = p.get_format_from_width(wf_sampwidth)
wf_samplerate = wf.getframerate()
print("Audio File Info:")
print("  Filename: %s" % filename)
print("  Channels: %d" % wf_channels)
print("  Sample Width: %d Bit" % (wf_sampwidth*8))
print("  Sample Rate: %d Hz" % wf_samplerate)
print("")

if dev_samplerate == wf_samplerate:
    print("Sample rate is equal, can't test conversion!")
    sys.exit(1)


print("Playing with device default output rate (%d Hz)..."
      % device['defaultSampleRate'])
stream = p.open(format=wf_format,
                channels=wf_channels,
                rate=dev_samplerate,
                output=True,
                output_device_index=dev_index)

data = wf.readframes(CHUNK)

cvstate = None
while data != '':
    newdata, cvstate = audioop.ratecv(
        data, wf_sampwidth, wf_channels, wf_samplerate,
        dev_samplerate, cvstate)
    stream.write(newdata)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()
print("DONE")
print("")

time.sleep(1)
print("Playing with file output rate (%d Hz)..." % wf.getframerate())
wf.rewind()
stream = p.open(format=wf_format,
                channels=wf_channels,
                rate=wf_samplerate,
                output=True,
                output_device_index=dev_index)

data = wf.readframes(CHUNK)

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()
print("DONE")

p.terminate()
