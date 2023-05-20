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
from tensorflow.keras.layers import ReLU, LeakyReLU

import sys, os
if __name__ == "__main__":
	sys.path.append(os.getcwd())


class CNN2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(CNN2DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

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
            self.regularizer = regularizers.l1(l=self.config['l1'])
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['l2'])
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['l1'], l2=self.config['l2'])
        else:
            self.regularizer = None

        # Loading the network architecture
        # [[Conv layer attributes], (norm + activation), 'pooling_type', [Conv layer attributes], (norm + activation), 'pooling_type']
        self.net_arc = self.config.get('network_architecture', [])
        self.net_list = []

        # Define the network architecture
        for idx, layer_attribute in enumerate(self.net_arc):
            if isinstance(layer_attribute, list): # Conv layer
                # Conv layer
                channel_size, kernel_size, strides, padding = layer_attribute
                self.net_list.append(Conv2D(channel_size, kernel_size, strides, padding, activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

                # normalization
                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list.append(LayerNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list.append(BatchNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")
                
                # activation
                if self.config.get('act_fn', 'relu') == 'relu':
                    self.net_list.append(ReLU())
                elif self.config.get('act_fn', 'relu') == 'lrelu':
                    self.net_list.append(LeakyReLU())
                else:
                    raise ValueError("Please use relu or leaky relu")

            elif isinstance(layer_attribute, str): # pooling type
                if layer_attribute == 'average':
                    self.net_list.append(AvgPool2D())
                elif layer_attribute == 'max':
                    self.net_list.append(MaxPooling2D())
                else:
                    raise ValueError("Please use average pooling or max pooling")

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, rows, cols, chennels)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        hidden = tf.reshape(hidden, (state.shape[0], -1))
        feature = self.feature(hidden)

        return feature


class AutoEncoder2DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(AutoEncoder2DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']
        self.ae_lr = self.config['lr']

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
            self.regularizer = regularizers.l1(l=self.config['l1'])
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['l2'])
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['l1'], l2=self.config['l2'])
        else:
            self.regularizer = None

        # Loading the network architecture
        # [[layer_type, Conv layer attributes], (norm + activation), 'pooling_type', [layer_type, Conv layer attributes], (norm + activation), 'pooling_type']
        self.net_arc = self.config.get('network_architecture', [])
        self.contract_net_list = []
        self.code_net, self.code_net_n = None, None
        self.expand_net_list = []

        # Define the network architecture
        # Contraction
        for idx, layer_attribute in enumerate(self.net_arc):
            if isinstance(layer_attribute, list): # Conv layer
                # Conv layer
                layer_type, channel_size, kernel_size, strides, padding = layer_attribute
                if layer_type == 'encoder':
                    self.net_list.append(Conv2D(channel_size, kernel_size, strides, padding, activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
                elif layer_type == 'decoder':
                    self.net_list.append(Conv2DTranspose(channel_size, kernel_size, strides, padding, activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

                # normalization
                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list.append(LayerNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list.append(BatchNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")
                
                # activation
                if self.config.get('act_fn', 'relu') == 'relu':
                    self.net_list.append(ReLU())
                elif self.config.get('act_fn', 'relu') == 'lrelu':
                    self.net_list.append(LeakyReLU())
                else:
                    raise ValueError("Please use relu or leaky relu")

            elif isinstance(layer_attribute, str): # pooling type
                if layer_attribute == 'average':
                    self.net_list.append(AvgPool2D())
                elif layer_attribute == 'max':
                    self.net_list.append(MaxPooling2D())
                else:
                    raise ValueError("Please use average pooling or max pooling")

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, rows, cols, chennels)
        '''
        # Contraction
        for idx, net in enumerate(self.contract_net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        # Code
        hidden = tf.reshape(hidden, (state.shape[0], -1))
        code = self.code_net(hidden)
        code = self.code_net_n(code)

        # Expansion
        for idx, net in enumerate(self.expand_net_list):
            if idx == 0:
                hidden = net(code)
            else:
                hidden = net(hidden)

        reconstruct = self.reconstruct(hidden)

        return code, reconstruct


class Inception2DExtractor(Model): # Todo
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(Inception2DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

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
        '''
        dim of state: (batch_size, rows, cols, chennels)
        '''
        feature = self.feature(state)

        return feature


class UNet2DExtractor(Model): # Todo
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(UNet2DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

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
        '''
        dim of state: (batch_size, rows, cols, chennels)
        '''
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
        if extractor_config["name"] == "AE2D":
            test, reconst = extractor(np.ones(shape=[128, 64, 64, 3]))
            print(f'shape of test tensor: {test.shape}')
            print(f'shape of reconst tensor: {reconst.shape}')
        else:
            test = extractor(np.ones(shape=[128, 64, 64, 3]))
            print(f'shape of test tensor: {test.shape}')
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
        test_ConvExtractor(Convolutional_Inception2d_extractor_config, Convolutional_Inception2d_feature_dim) # Todo
    elif test_switch == 4:
        test_ConvExtractor(Convolutional_UNet2d_extractor_config, Convolutional_UNet2d_feature_dim) # Todo
    else:
        raise ValueError("Please correct the test switch in [1, 2, 3, 4]")
