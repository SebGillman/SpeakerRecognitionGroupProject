import numpy as np
import sounddevice as sd
import wave

class RecordAudio:
    def __init__(self):
        # record parameters
        self.API_KEY = ""
        self.wave_length = 3
        self.sample_rate = 44100

        # self.frames = []
        self.keep_going = True

    def record(self, output_path="audio_db/temp.wav"):
        """
        Terms meaning
        :param output_path: path of save recording, with file format wav
        :param record_seconds: record time, default setting will be 3s
        :return: file path of the audio recordings
        """

        print("[RECORDER] Listening ...")
        # Start recording (wave_length Record for seconds. Wait until the recording is finished with wait)
        data = sd.rec(int(self.wave_length * self.sample_rate), self.sample_rate, channels=1)
        sd.wait()
        print("[RECORDER] Recording finished")

        # Normalize. Since it is recorded with 16 bits of quantization bit, it is maximized in the range of int16.
        data = data / data.max() * np.iinfo(np.int16).max

        # float -> int
        data = data.astype(np.int16)

        print("[RECORDER] Saving ...")
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(1)  # monaural
        wf.setsampwidth(2)  # 16bit=2byte
        wf.setframerate(self.sample_rate)
        wf.writeframes(data.tobytes())  # Convert to byte string
        wf.close()
        return output_path

