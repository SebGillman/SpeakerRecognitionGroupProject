import random
import sys

import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore")

import librosa
import numpy as np
from torch.utils import data


# load and preprocess audio
def load_audio(audio_path, feature_method='melspectrogram', mode='train', sr=16000, chunk_duration=3, augmentors=None):
    # load audio
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    if mode == 'train':
        # random cut
        num_wav_samples = wav.shape[0]
        # data too short not good for training
        if num_wav_samples < sr:
            raise Exception(f'audio length is less than 1s, the actual length is：{(num_wav_samples/sr):.2f}s')
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 4)] = 0
                wav = wav[:-random.randint(1, sr // 4)]
        # enhance data
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug':continue
                wav = augmentor(wav)
    elif mode == 'eval':
        # only use certain length in case of overflow the size
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
    if feature_method == 'melspectrogram':
        # cal mel
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    elif feature_method == 'spectrogram':
        # cal non-mel
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'proprocess method {feature_method} 不存在！')
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    # enhance data
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    return features



class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, feature_method='melspectrogram', mode='train', sr=16000, chunk_duration=3, augmentors=None):
        super(CustomDataset, self).__init__()
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            features = load_audio(audio_path, feature_method=self.feature_method, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'proprocess method {self.feature_method} 不存在！')


# process data in a unit of batch
def collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    input_lens = []
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        inputs[x, :, :seq_length] = tensor[:, :]
        input_lens.append(seq_length/max_audio_length)
    input_lens = np.array(input_lens, dtype='float32')
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens)
