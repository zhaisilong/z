from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Metrics(object):

    @staticmethod
    def remap(x: np.array, x_min, x_max):
        assert x_max > x_min, '最大最小值一样'
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def norm(scores: np.array, epsilon=0.0):
        return np.clip(Metrics.remap(scores, scores.min(), scores.max()), epsilon, 1.0)
