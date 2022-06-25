import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import librosa


class NoisePerturbAugmentor(object):


    def __init__(self, min_snr_dB=10, max_snr_dB=30, noise_path="dataset/noise", sr=16000, prob=0.5):
        self.prob = prob
        self.sr = sr
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._noise_files = self.get_noise_file(noise_path=noise_path)

    # obtain all noise data
    @staticmethod
    def get_noise_file(noise_path):
        noise_files = []
        if not os.path.exists(noise_path): return noise_files
        for file in os.listdir(noise_path):
            noise_files.append(os.path.join(noise_path, file))
        return noise_files

    @staticmethod
    def rms_db(wav):
        """
        :rtype: float
        """
        mean_square = np.mean(wav ** 2)
        return 10 * np.log10(mean_square)

    def __call__(self, wav):
        """add background noise audio files

        :param wav: librosa read loaded data
        :type wav: ndarray
        """
        if random.random() > self.prob: return wav
        if len(self._noise_files) == 0: return wav
        noise, r = librosa.load(random.choice(self._noise_files), sr=self.sr)
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
        noise_gain_db = min(self.rms_db(wav) - self.rms_db(noise) - snr_dB, 300)
        noise *= 10. ** (noise_gain_db / 20.)
        noise_new = np.zeros(wav.shape, dtype=np.float32)
        if noise.shape[0] >= wav.shape[0]:
            start = random.randint(0, noise.shape[0] - wav.shape[0])
            noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]
        else:
            start = random.randint(0, wav.shape[0] - noise.shape[0])
            noise_new[start:start + noise.shape[0]] = noise[:]
        wav += noise_new
        return wav
