from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LayerNormalization


# Project 1
class ResidualExtractor(Model):
    def __init__(self, )-> None:
        super(ResidualExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class AutoEncoderExtractor(Model):
    def __init__(self, )-> None:
        super(AutoEncoderExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class UNetExtractor(Model):
    def __init__(self, )-> None:
        super(AutoEncoderExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


# Project 2
class LSTMExtractor(Model):
    def __init__(self, )-> None:
        super(LSTMExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class CNN1DExtractor(Model):
    def __init__(self, )-> None:
        super(CNN1DExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class BiLSTMExtractor(Model):
    def __init__(self, )-> None:
        super(BiLSTMExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


# Project 3
class AttentionExtractor(Model):
    def __init__(self, )-> None:
        super(AttentionExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class TransDuctiveGNNExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class InductiveGNNExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class TransformerExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


if __name__ == "__main__":
    pass