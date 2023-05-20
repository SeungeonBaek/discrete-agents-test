import traceback

from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN as RNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AvgPool1D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU

import sys, os
if __name__ == "__main__":
	sys.path.append(os.getcwd())


class RNNExtractor(Model):
    """
    RNN extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: hidden units of each layer in whole neural network. ex) [256, 256]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function between each layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(RNNExtractor,self).__init__()

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
        self.net_arc = self.config.get('network_architecture', [])
        if self.config.get('use_norm', False) == True:
            self.net_list = [None for _ in self.net_arc*2]
        else:
            self.net_list = [None for _ in self.net_arc]

        # Define the network architecture
        if self.config.get('use_norm', False) == True:
            for idx in range(0, len(self.net_arc*2)-2, 2):
                self.net_list[idx] = RNN(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)

                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list[idx+1] = LayerNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list[idx+1] = BatchNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

            self.net_list[-2] = RNN(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

            if self.config.get('norm_type', None) == "layer_norm":
                self.net_list[-1] = LayerNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "batch_norm":
                self.net_list[-1] = BatchNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "group_norm":
                raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
            else:
                raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        else:
            for idx in range(len(self.net_arc)-1):
                self.net_list[idx] = RNN(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)
            
            self.net_list[-1] = RNN(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, time_window, features)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


class LSTMExtractor(Model):
    """
    LSTM extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: hidden units of each layer in whole neural network. ex) [256, 256]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function between each layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(LSTMExtractor,self).__init__()

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
        self.net_arc = self.config.get('network_architecture', [])
        if self.config.get('use_norm', False) == True:
            self.net_list = [None for _ in self.net_arc*2]
        else:
            self.net_list = [None for _ in self.net_arc]

        # Define the network architecture
        if self.config.get('use_norm', False) == True:
            for idx in range(0, len(self.net_arc*2)-2, 2):
                self.net_list[idx] = LSTM(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)

                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list[idx+1] = LayerNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list[idx+1] = BatchNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

            self.net_list[-2] = LSTM(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

            if self.config.get('norm_type', None) == "layer_norm":
                self.net_list[-1] = LayerNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "batch_norm":
                self.net_list[-1] = BatchNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "group_norm":
                raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
            else:
                raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        else:
            for idx in range(len(self.net_arc)-1):
                self.net_list[idx] = LSTM(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)
            
            self.net_list[-1] = LSTM(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, time_window, features)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


class GRUExtractor(Model):
    """
    GRU extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: hidden units of each layer in whole neural network. ex) [256, 256]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function between each layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GRUExtractor,self).__init__()

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
        self.net_arc = self.config.get('network_architecture', [])
        if self.config.get('use_norm', False) == True:
            self.net_list = [None for _ in self.net_arc*2]
        else:
            self.net_list = [None for _ in self.net_arc]

        # Define the network architecture
        if self.config.get('use_norm', False) == True:
            for idx in range(0, len(self.net_arc*2)-2, 2):
                self.net_list[idx] = GRU(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)

                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list[idx+1] = LayerNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list[idx+1] = BatchNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

            self.net_list[-2] = GRU(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

            if self.config.get('norm_type', None) == "layer_norm":
                self.net_list[-1] = LayerNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "batch_norm":
                self.net_list[-1] = BatchNormalization(axis=-1)
            elif self.config.get('norm_type', None) == "group_norm":
                raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
            else:
                raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        else:
            for idx in range(len(self.net_arc)-1):
                self.net_list[idx] = GRU(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=True)
            
            self.net_list[-1] = GRU(self.net_arc[idx], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, return_sequences=False)

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, time_window, features)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


class CNN1DExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(CNN1DExtractor,self).__init__()

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
        # [[Conv layer attributes], (norm + activation), [Conv layer attributes], (norm + activation), 'pooling_type']
        # [[Conv layer attributes], (norm + activation), 'pooling_type', [Conv layer attributes], (norm + activation), 'pooling_type']
        self.net_arc = self.config.get('network_architecture', [])
        self.net_list = []

        # Define the network architecture
        for idx, layer_attribute in enumerate(self.net_arc):
            if isinstance(layer_attribute, list): # Conv layer
                # Conv layer
                channel_size, kernel_size, strides, padding = layer_attribute
                self.net_list.append(Conv1D(channel_size, kernel_size, strides, padding, activation=None, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

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
                    self.net_list.append(AvgPool1D())
                elif layer_attribute == 'max':
                    self.net_list.append(MaxPooling1D())
                else:
                    raise ValueError("Please use average pooling or max pooling")

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, time_window, features)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        hidden = tf.reshape(hidden, (state.shape[0], -1))
        feature = self.feature(hidden)

        return feature


def load_RecurExtractor(extractor_config:Dict, feature_dim)-> Model:
    if extractor_config.get('name', None) == 'RNN':
        return RNNExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'LSTM':
        return LSTMExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GRU':
        return GRUExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'CNN1D':
        return CNN1DExtractor(extractor_config, feature_dim)
    
    else:
        raise ValueError("please use the MLPExtractor in ['RNN', 'LSTM', 'GRU', 'CNN1D']")


def test_RecurExtractor(extractor_config:Dict, feature_dim:int)-> None:
    extractor = load_RecurExtractor(extractor_config, feature_dim)

    try:
        test = extractor(np.ones(shape=[128, 4, 128]))
        print(f'shape of test tensor: {test.shape}')
    except Exception as e:
        print(f"error: {e}")
        print(f"error: {traceback.format_exc()}")


if __name__ == "__main__":
    from extractor_config import REC_RNN_extractor_config, REC_RNN_feature_dim
    from extractor_config import REC_LSTM_extractor_config, REC_LSTM_feature_dim
    from extractor_config import REC_GRU_extractor_config, REC_GRU_feature_dim
    from extractor_config import REC_CNN1d_extractor_config, REC_CNN1d_feature_dim

    """
    Recurrent Extractor
    1: RNN Extractor, 2: LSTM Extractor
    3: GRU Extractor, 4: CNN1D Extractor
    """

    test_switch = 4

    # Test any extractor
    if test_switch == 1:
        test_RecurExtractor(REC_RNN_extractor_config, REC_RNN_feature_dim)
    elif test_switch == 2:
        test_RecurExtractor(REC_LSTM_extractor_config, REC_LSTM_feature_dim)
    elif test_switch == 3:
        test_RecurExtractor(REC_GRU_extractor_config, REC_GRU_feature_dim)
    elif test_switch == 4:
        test_RecurExtractor(REC_CNN1d_extractor_config, REC_CNN1d_feature_dim)
    else:
        raise ValueError("Please correct the test switch in [1, 2, 3, 4]")