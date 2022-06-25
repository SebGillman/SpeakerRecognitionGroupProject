import random


class VolumePerturbAugmentor(object):
    """add random volume

    :param min_gain_dBFS:  least gain
    :type min_gain_dBFS: int
    :param max_gain_dBFS:  max least gain
    :type max_gain_dBFS: int
    :type prob: float
    """

    def __init__(self, min_gain_dBFS=-15, max_gain_dBFS=15, prob=0.5):
        self.prob = prob
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS

    def __call__(self, wav):
        """change volume

        :param wav: librosa read data
        :type wav: ndarray
        """
        if random.random() > self.prob: return wav
        gain = random.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        wav *= 10.**(gain / 20.)
        return wav
