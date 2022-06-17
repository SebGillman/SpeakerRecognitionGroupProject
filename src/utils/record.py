import wave

import pyaudio
from tqdm import tqdm

class RecordAudio:
    def __init__(self):
        # record parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # open audio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def record(self, record_seconds=3, cloud = False):
        """
        Terms meaning
        :param output_path: path of save recording, with file format wav
        :param record_seconds: record time, default setting will be 3s
        :return: file path of the audio recordings
        """
        if cloud:
            output_path="tmp/temp.wav"
        else:
            output_path="audio_db/temp.wav"

        i = input("\nPress ENTER to start recording (recording time is 3 seconds) ")
        print("\n[RECORDER] Listening ......")
        frames = []
        for i in tqdm(range(0, int(self.rate / self.chunk * record_seconds))):
            data = self.stream.read(self.chunk,exception_on_overflow = False)
            frames.append(data)

        print("[RECORDER] Recording finished!")
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return output_path

    def recordconst(self,  record_seconds=2.5, cloud = False):
        """
        Terms meaning
        :param output_path: path of save recording, with file format wav
        :param record_seconds: record time, default setting will be 3s
        :return: file path of the audio recordings
        """
        if cloud:
            output_path="tmp/temp.wav"
        else:
            output_path="audio_db/temp.wav"

        frames = []
        for i in range(0, int(self.rate / self.chunk * record_seconds)):
            data = self.stream.read(self.chunk, exception_on_overflow = False)
            frames.append(data)

        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return output_path

