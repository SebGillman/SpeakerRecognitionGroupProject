import wave

import pyaudio
from tqdm import tqdm

class RecordAudio:
    def __init__(self):
        # record parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100

        # open audio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def record(self, output_path="audio_db/temp.wav", record_seconds=3):
        """
        Terms meaning
        :param output_path: path of save recording, with file format wav
        :param record_seconds: record time, default setting will be 3s
        :return: file path of the audio recordings
        """
        i = input("press ENTER to start recording (recording time is 3 seconds) ")
        print("[RECORDER] Listening ......")
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

    def recordconst(self, output_path="audio_db/temp.wav", record_seconds=0.5):
        """
        Terms meaning
        :param output_path: path of save recording, with file format wav
        :param record_seconds: record time, default setting will be 3s
        :return: file path of the audio recordings
        """
        print("[RECORDER] Listening ......")
        frames = []
        for i in range(0, int(self.rate / self.chunk * record_seconds)):
            data = self.stream.read(self.chunk, exception_on_overflow = False)
            frames.append(data)

        print("[RECORDER] Recording finished!")
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return output_path

