import traceback

from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, AvgPool2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GroupNormalization

import sys, os
if __name__ == "__main__":
	sys.path.append(os.getcwd())


class CNN2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(CNN2DExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class AutoEncoder2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(AutoEncoder2DExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class Inception2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(Inception2DExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class UNet2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(UNet2DExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


def load_ConvExtractor(extractor_config:Dict, feature_dim):
    if extractor_config.get('name', None) == 'CNN2D':
        return CNN2DExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'AutoEncoder2D':
        return AutoEncoder2DExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'Inception2D':
        return Inception2DExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'UNet2D':
        return UNet2DExtractor(extractor_config, feature_dim)

    else:
        raise ValueError("please use the MLPExtractor in ['CNN2D', 'AutoEncoder2D', 'Inception2D', 'UNet2D']")


def test_ConvExtractor(extractor_config:Dict, feature_dim:int)-> None: # Todo
    extractor = load_ConvExtractor(extractor_config, feature_dim)

    try:
        test = extractor(np.ones(shape=[128, 128]))
    except Exception as e:
        print(f"error: {e}")
        print(f"error: {traceback.format_exc()}")


if __name__ == "__main__": # Todo
    from extractor_config import Convolutional_CNN2d_extractor_config, Convolutional_CNN2d_feature_dim
    from extractor_config import Convolutional_AE2d_extractor_config, Convolutional_AE2d_feature_dim
    from extractor_config import Convolutional_Inception2d_extractor_config, Convolutional_Inception2d_feature_dim
    from extractor_config import Convolutional_UNet2d_extractor_config, Convolutional_UNet2d_feature_dim

    """
    Convolutional Extractor
    1: CNN2D Extractor,       2: AE2D Extractor,
    3: Inception2D Extractor  4: Unet2D Extractor
    """

    test_switch = 2

    # Test any extractor
    if test_switch == 1:
        test_ConvExtractor(Convolutional_CNN2d_extractor_config, Convolutional_CNN2d_feature_dim)
    elif test_switch == 2:
        test_ConvExtractor(Convolutional_AE2d_extractor_config, Convolutional_AE2d_feature_dim)
    elif test_switch == 3:
        test_ConvExtractor(Convolutional_Inception2d_extractor_config, Convolutional_Inception2d_feature_dim)
    elif test_switch == 4:
        test_ConvExtractor(Convolutional_UNet2d_extractor_config, Convolutional_UNet2d_feature_dim)
    else:
        raise ValueError("Please correct the test switch in [1, 2, 3, 4]")
